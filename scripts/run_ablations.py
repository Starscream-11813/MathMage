"""Run A1-A8 and C1-C2; defaults to mock mode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathmage.experiments.runner import main
if __name__ == "__main__":
    main(["ablations", *sys.argv[1:]])
