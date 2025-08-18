import os
from pathlib import Path

output_folder: Path = Path("tests/output")
output_folder.mkdir(exist_ok=True, parents=True)

tmp_folder: Path = Path("tests/tmp")
tmp_folder.mkdir(exist_ok=True, parents=True)

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"
