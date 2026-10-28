"""Anonymize the repo for public release.

1. Remove the tests/ folder and the launch_*.sh scripts.
2. Remove local/ folders and any checkpoints in the repo.
3. Replace all gpfs and home-directory paths with "/path/to".

Note: the patterns are built dynamically so this script does not
rewrite itself when run.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REMOVE_DIR_NAMES = {
    "tests", "local",
    "checkpoints", "checkpoint", "ckpt", "ckpts",
    "logs", "wandb", "mlruns", "outputs", "output",
}
REMOVE_FILE_SUFFIXES = {".ckpt", ".pt", ".pth", ".pkl"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".idea", ".pytest_cache"}
TEXT_EXTENSIONS = {
    ".py", ".md", ".rst", ".yaml", ".yml", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".txt", ".csv", ".ipynb", ".def", ".xml",
}
# built dynamically so this file doesn't match its own pattern
GPFS_PATTERN = re.compile("/" + "gpfs" + r"[^\s\"'\]\),;]*")
HOME_PATTERN = re.compile(r"/(home|Users)/[^\s\"'\]\),;]*")


def is_skipped(path):
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def remove_paths():
    for hidden in (ROOT / ".idea", ROOT / ".pytest_cache"):
        if hidden.is_dir():
            shutil.rmtree(hidden)
            print(f"removed dir:  {hidden.relative_to(ROOT)}")
    for path in sorted(ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if is_skipped(path) or not path.exists():
            continue
        name = path.name.lower()
        try:
            if path.is_dir():
                if name in REMOVE_DIR_NAMES:
                    shutil.rmtree(path)
                    print(f"removed dir:  {path.relative_to(ROOT)}")
            else:
                if (
                    path.suffix.lower() in REMOVE_FILE_SUFFIXES
                    or ".ckpt" in name
                    or name.startswith("launch_")
                    or name == "launch.sh"
                ):
                    path.unlink()
                    print(f"removed file: {path.relative_to(ROOT)}")
        except OSError as e:
            print(f"skipped {path.relative_to(ROOT)}: {e}")


def anonymize_files():
    for path in ROOT.rglob("*"):
        if is_skipped(path) or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_EXTENSIONS and not path.name.startswith("Dockerfile"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        new_text = HOME_PATTERN.sub("/path/to", GPFS_PATTERN.sub("/path/to", text))
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"anonymized: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    remove_paths()
    anonymize_files()
    print("done.")
