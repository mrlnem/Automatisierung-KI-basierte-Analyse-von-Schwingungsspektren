import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .data_load import discover_csvs, load_signals, extract_vibration_xyz
from .features_stft import to_rgb_image
from .model_cnn import SimpleCNN
from .config import Config


def label_from_path(path: Path, cfg: Config) -> int:
    """
    Derive label (0/1/2) from the CSV filename or parent folder name.
    PHM2010 filenames encode wear state, e.g.:
      - c1_001.csv -> "uniform" (early wear, small index)
      - c1_315.csv -> "severe" or "rapid" (late wear, high index)
    Convention used here: last numeric part of stem indicates
    wear stage; map via Config.classes index.
    Falls back to 0 (uniform/OK) if no match is found.
    """
    name = path.stem.lower()
    for i, cls in enumerate(cfg.classes):
        if cls in name:
            return i
    # PHM2010: wear progression by file index (approx. thirds)
    parts = name.split("_")
    try:
        idx = int(parts[-1])
        if idx >= 200:
            return 2  # rapid
        elif idx >= 100:
            return 1  # severe
        else:
            return 0  # uniform
    except ValueError:
        return 0


class SpectroDataset(Dataset):
    def __init__(self, refs, cfg):
        self.refs = refs
        self.cfg = cfg

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        path, _ = self.refs[idx]

        # Try a few files until we find a usable signal CSV
        for _ in range(10):
            df = load_signals(path)
            try:
                vx, vy, vz = extract_vibration_xyz(df)
                img = to_rgb_image(vx, vy, vz, self.cfg)
                x = torch.tensor(img).permute(2, 0, 1).float()
                y = label_from_path(path, self.cfg)  # Real label from filename
                return x, y
            except Exception:
                # pick next file
                idx = (idx + 1) % len(self.refs)
                path, _ = self.refs[idx]

        # If we fail too often, crash with a clear error
        raise RuntimeError("Could not find a usable CSV with >=3 numeric columns for vibration channels.")


def train_model(
    data_root: Path,
    out_path: Path,
    cfg: Config,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
):
    refs = discover_csvs(data_root, cfg.train_cutters)
    ds = SpectroDataset(refs, cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SimpleCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print(f"Epoch {ep+1} done")

    out_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_path)
