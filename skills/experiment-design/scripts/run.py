"""Run the experiment-design workflow using the shared scholar-daily runtime."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if (ROOT / "scholar_daily").is_dir():
    sys.path.insert(0, str(ROOT))

from scholar_daily.skill_runner import main

if __name__ == "__main__":
    raise SystemExit(main("experiment-design"))
