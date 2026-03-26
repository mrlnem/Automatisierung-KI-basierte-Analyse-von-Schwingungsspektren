from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .data_download import download_phm2010
from .train import train_model
from .inference import predict


def main():
    cfg = Config()

    parser = argparse.ArgumentParser(
        prog="Automatisierung KI-basierte Analyse",
        description="STFT-Spektrogramme -> CNN -> Nudging (OK/DEFECT)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- train command ----
    p_train = sub.add_parser("train", help="Train CNN model on PHM2010")
    p_train.add_argument("--out", type=str, required=True, help="Output model path, e.g. models/stft_cnn.pt")
    p_train.add_argument("--data", type=str, default=None, help="Optional data root. If omitted, downloads via kagglehub.")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)

    # ---- predict command ----
    p_pred = sub.add_parser("predict", help="Run inference + nudging")
    p_pred.add_argument("--model", type=str, required=True, help="Path to trained model")
    p_pred.add_argument("--data", type=str, default=None, help="Optional data root. If omitted, downloads via kagglehub.")
    p_pred.add_argument("--split", type=str, default="test", choices=["train", "test"])

    args = parser.parse_args()

    # Determine data root (auto download if not provided)
    data_root = Path(args.data) if args.data else download_phm2010()

    if args.cmd == "train":
        train_model(
            data_root=data_root,
            out_path=Path(args.out),
            cfg=cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.cmd == "predict":
        predict(
            data_root=data_root,
            model_path=Path(args.model),
            split=args.split,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
