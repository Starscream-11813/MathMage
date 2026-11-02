"""Randomized Z3/Z6 continuations from sealed checkpoints; defaults to mock mode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathmage.experiments.mechanisms import main
if __name__ == "__main__":
    main()
