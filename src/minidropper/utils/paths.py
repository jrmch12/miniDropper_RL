from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve(path) -> Path:
    """Absolute paths stay as they are; relative paths are taken from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
