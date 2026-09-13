"""Repo-root shim: `python pipeline.py --config config/config.yaml --mode predict`.

Adds src/ to sys.path and delegates to project_listen.pipeline.main().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from project_listen.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
