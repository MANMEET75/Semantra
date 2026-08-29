"""Build PyPI wheels without putting both large models in one wheel."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run_build(project: Path, output: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(output)],
        cwd=project,
        check=True,
    )


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    with tempfile.TemporaryDirectory(prefix="semantra-release-") as temp:
        staging = Path(temp) / "semantra"
        shutil.copytree(
            ROOT,
            staging,
            ignore=shutil.ignore_patterns(".git", ".venv", ".uv-cache", "build", "dist"),
        )
        shutil.rmtree(staging / "src" / "semantra" / "assets" / "multilingual-e5-small")
        run_build(staging, DIST)
    run_build(ROOT / "packages" / "semantra-multilingual", DIST)


if __name__ == "__main__":
    main()
