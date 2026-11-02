"""Build frozen training artifacts. Defaults to mock mode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathmage.experiments.offline import main
if __name__ == "__main__":
    main()
