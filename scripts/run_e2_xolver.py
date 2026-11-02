"""Run MathMage on the Xolver comparison suite; defaults to mock mode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathmage.experiments.runner import main
if __name__ == "__main__":
    main(["e2", *sys.argv[1:]])
