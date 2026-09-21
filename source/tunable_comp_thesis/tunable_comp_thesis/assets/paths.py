from pathlib import Path

# assets/ -> my_project/ (package) -> source/my_project/ -> data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"