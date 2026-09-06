"""Compatibility command for the simulator's decode extrapolation method."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.decode_batch_estimate import *

if __name__ == "__main__":
    main()
