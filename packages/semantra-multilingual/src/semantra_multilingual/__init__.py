"""Bundled multilingual model assets for Semantra."""

from pathlib import Path


def get_asset_dir() -> str:
    """Return the installed multilingual model asset directory."""
    return str(Path(__file__).parent / "assets" / "multilingual-e5-small")


__all__ = ["get_asset_dir"]
