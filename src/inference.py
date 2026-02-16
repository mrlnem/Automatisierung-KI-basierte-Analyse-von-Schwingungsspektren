from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .config import Config
from .data_load import discover_csvs, load_signals, extract_vibration_xyz
from .decision import decide_nudge
from .features_stft import to_rgb_image
from .model_cnn import SimpleCNN


def predict(data_root: Path, model_path: Path, split: str, cfg: Config):
    cutters = cfg.test_cutters if split == "test" else cfg.train_cutters
    refs = discover_csvs(data_root, cutters)

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    rows = []
    skipped = 0

    for path, cutter in refs:
        try:
            df = load_signals(path)
            vx, vy, vz = extract_vibration_xyz(df)
            img = to_rgb_image(vx, vy, vz, cfg)
        except Exception as e:
            skipped += 1
            # Uncomment for debugging:
            # print(f"Skipping {path} ({e})")
            continue

        x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        probs = torch.softmax(model(x), dim=1).detach().numpy()[0]

        label, nudge, emoji = decide_nudge(probs)
        print(f"{emoji} {nudge} | {cutter} | {path.name}")

        rows.append({
            "file": str(path),
            "cutter": cutter,
            "p_uniform": float(probs[0]),
            "p_severe": float(probs[1]),
            "p_rapid": float(probs[2]),
            "label": label,
            "nudge": nudge,
            "emoji": emoji,
        })

    cfg.results_dir.mkdir(exist_ok=True)
    df_out = pd.DataFrame(rows)
    pred_path = cfg.results_dir / "predictions.csv"
    df_out.to_csv(pred_path, index=False)

    # ---- Summary JSON ----
    ok = int((df_out["nudge"] == "OK").sum())
    defect = int((df_out["nudge"] == "DEFECT").sum())
    total = int(len(df_out))

    summary = {
        "split": split,
        "total_predicted": total,
        "ok": ok,
        "defect": defect,
        "ok_rate": ok / max(1, total),
        "skipped_files": skipped,
    }

    with open(cfg.results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Human-readable report ----
    with open(cfg.results_dir / "report.md", "w") as f:
        f.write(f"# Prediction Report ({split})\n\n")
        f.write(f"- Predicted files: {total}\n")
        f.write(f"- Skipped files: {skipped}\n")
        f.write(f"- 🟢 OK: {ok}\n")
        f.write(f"- 🔴 DEFECT: {defect}\n")
        f.write(f"- OK rate: {summary['ok_rate']:.2%}\n")

    # ---- Optional: aggregation per cutter ("Teil" = Cutter) ----
    if total > 0:
        part = (
            df_out.groupby("cutter")["nudge"]
            .apply(lambda s: "DEFECT" if (s == "DEFECT").any() else "OK")
            .reset_index()
            .rename(columns={"nudge": "part_nudge"})
        )
        part.to_csv(cfg.results_dir / "part_summary.csv", index=False)

    print("\nSaved:")
    print(f"- {pred_path}")
    print(f"- {cfg.results_dir / 'summary.json'}")
    print(f"- {cfg.results_dir / 'report.md'}")
    if total > 0:
        print(f"- {cfg.results_dir / 'part_summary.csv'}")
