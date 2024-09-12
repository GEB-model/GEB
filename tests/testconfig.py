import os
import pathlib

output_folder = pathlib.Path("tests/output")
output_folder.mkdir(exist_ok=True)

tmp_folder = pathlib.Path("tests/tmp")
tmp_folder.mkdir(exist_ok=True)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
