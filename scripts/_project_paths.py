"""Shared filesystem locations for the reproducible analysis pipeline."""

from pathlib import Path


# ``scripts`` is stored directly under the analysis root (``final``).
FINAL_DIR = Path(__file__).resolve().parents[1]
