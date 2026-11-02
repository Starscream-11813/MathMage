"""Run the default-offline pilot; see --help for a reviewed live plan."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathmage.experiments.runner import main
if __name__ == "__main__":
    main(["pilot", *sys.argv[1:]])
