"""Version definition for GEB."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib

            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                if "project" in data and "version" in data["project"]:
                    return data["project"]["version"]
        except Exception:
            pass
    try:
        return version("geb")
    except PackageNotFoundError:
        return "unknown"


__version__: str = _get_version()
