from click.testing import CliRunner
from geb.cli import build

runner = CliRunner()


def test_build():
    result = runner.invoke(build, ["-wd", "examples"])

    if result.exit_code != 0:
        error_message = (
            result.exception.strerror
            if hasattr(result.exception, "strerror")
            else result.output
        )
        raise AssertionError(f"Build failed: {error_message}")
