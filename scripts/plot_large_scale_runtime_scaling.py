"""Plot large-scale spinup runtime scaling from GEB timing logs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

DEFAULT_MODELS_DIR = Path("/gpfs/work5/0/prjs2035/GEB/models/large_scale_runtimes")
DEFAULT_OUTPUT_PATH = Path(".benchmarks/large_scale_spinup_runtime_scaling.png")
DEFAULT_CSV_PATH = Path(".benchmarks/large_scale_spinup_runtime_scaling.csv")
STEP_TIME_PATTERN = re.compile(
    r"step\s+\d{4}-\d{2}-\d{2}\s+took\s+(?P<seconds>[0-9.]+)s"
)
CPU_COUNT_PATTERN = re.compile(r"spinup_(?P<cpus>\d+)_?CPUs\.log$")


@dataclass(frozen=True)
class RuntimePoint:
    """Runtime benchmark point for one region and CPU count.

    Args:
        region_name: Region identifier, e.g. ``Europe_004``.
        cpu_count: Number of CPUs requested for the spinup run.
    average_timestep_seconds: Average runtime of the final two timesteps
        (seconds).
        basin_area_km2: Total basin area (km²).
        log_path: Path to the parsed spinup log.

    """

    region_name: str
    cpu_count: int
    average_timestep_seconds: float
    basin_area_km2: float
    log_path: Path


def parse_timestep_seconds(log_path: Path) -> list[float]:
    """Extract per-timestep runtimes from a spinup log.

    Args:
        log_path: Path to the spinup log.

    Returns:
        Per-timestep runtimes (seconds), in log order.

    Raises:
        FileNotFoundError: If ``log_path`` does not exist.
        ValueError: If the log does not contain timestep timing lines.

    """
    if not log_path.exists():
        raise FileNotFoundError(f"Spinup log does not exist: {log_path}")

    log_text: str = log_path.read_text(errors="replace")
    timestep_seconds: list[float] = [
        float(match.group("seconds")) for match in STEP_TIME_PATTERN.finditer(log_text)
    ]
    if not timestep_seconds:
        raise ValueError(f"No timestep timing lines found in {log_path}")
    return timestep_seconds


def average_final_timesteps(log_path: Path, timestep_count: int = 2) -> float:
    """Average the final timesteps in a spinup log.

    Args:
        log_path: Path to the spinup log.
        timestep_count: Number of final timesteps to average. Must be positive.

    Returns:
        Average runtime of the final timesteps (seconds).

    Raises:
        ValueError: If ``timestep_count`` is invalid or the log has too few
            timestep timing lines.

    """
    if timestep_count <= 0:
        raise ValueError(f"timestep_count must be positive, got {timestep_count}")

    timestep_seconds: list[float] = parse_timestep_seconds(log_path)
    if len(timestep_seconds) < timestep_count:
        raise ValueError(
            f"Need at least {timestep_count} timestep timing lines in {log_path}, "
            f"found {len(timestep_seconds)}"
        )

    final_timesteps: list[float] = timestep_seconds[-timestep_count:]
    return sum(final_timesteps) / len(final_timesteps)


def read_basin_area_km2(model_config_path: Path) -> float:
    """Read the total basin area from a region model configuration.

    Args:
        model_config_path: Path to the region ``model.yml`` file.

    Returns:
        Total basin area (km²).

    Raises:
        FileNotFoundError: If ``model_config_path`` does not exist.
        ValueError: If the basin area is missing or invalid.

    """
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config does not exist: {model_config_path}")

    with model_config_path.open() as config_file:
        model_config: dict[str, object] = yaml.safe_load(config_file) or {}

    basin_config = model_config.get("basin")
    if not isinstance(basin_config, dict):
        raise ValueError(f"Missing basin section in {model_config_path}")

    raw_area_km2 = basin_config.get("total_area_km2")
    try:
        basin_area_km2 = float(raw_area_km2)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid basin.total_area_km2 in {model_config_path}: {raw_area_km2!r}"
        ) from error

    if basin_area_km2 <= 0:
        raise ValueError(
            f"basin.total_area_km2 must be positive in {model_config_path}, "
            f"got {basin_area_km2}"
        )
    return basin_area_km2


def collect_region_points(
    models_dir: Path,
    region_name: str,
    timestep_count: int,
) -> list[RuntimePoint]:
    """Collect runtime benchmark points for one region.

    Args:
        models_dir: Directory containing region folders.
        region_name: Region identifier to parse.
        timestep_count: Number of final timesteps to average.

    Returns:
        Runtime points sorted by CPU count.

    Raises:
        FileNotFoundError: If no matching spinup logs are found.
        ValueError: If a matching log filename does not contain a CPU count.

    """
    logs_dir: Path = models_dir / region_name / "base" / "logs"
    basin_area_km2: float = read_basin_area_km2(
        models_dir / region_name / "base" / "model.yml"
    )
    log_paths: list[Path] = sorted(logs_dir.glob("spinup_*CPU*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No spinup_*CPUs.log files found in {logs_dir}")

    points: list[RuntimePoint] = []
    for log_path in log_paths:
        cpu_match: re.Match[str] | None = CPU_COUNT_PATTERN.search(log_path.name)
        if cpu_match is None:
            raise ValueError(f"Could not parse CPU count from {log_path.name}")

        cpu_count: int = int(cpu_match.group("cpus"))
        average_timestep_seconds: float = average_final_timesteps(
            log_path=log_path,
            timestep_count=timestep_count,
        )
        points.append(
            RuntimePoint(
                region_name=region_name,
                cpu_count=cpu_count,
                average_timestep_seconds=average_timestep_seconds,
                basin_area_km2=basin_area_km2,
                log_path=log_path,
            )
        )

    return sorted(points, key=lambda point: point.cpu_count)


def write_runtime_csv(points: list[RuntimePoint], csv_path: Path) -> None:
    """Write parsed runtime benchmark points to CSV.

    Args:
        points: Runtime benchmark points.
        csv_path: Output CSV path.

    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "region",
                "basin_area_km2",
                "cpus",
                "average_final_timestep_seconds",
                "log_path",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.region_name,
                    f"{point.basin_area_km2:.2f}",
                    point.cpu_count,
                    f"{point.average_timestep_seconds:.6f}",
                    point.log_path,
                ]
            )


def plot_runtime_scaling(
    small_model_points: list[RuntimePoint],
    large_model_points: list[RuntimePoint],
    output_path: Path,
) -> None:
    """Plot runtime scaling for small and large model regions.

    Args:
        small_model_points: Runtime points for the small model region.
        large_model_points: Runtime points for the large model region.
        output_path: Output figure path.

    Raises:
        ValueError: If either model has no runtime points.

    """
    if not small_model_points:
        raise ValueError("small_model_points must not be empty")
    if not large_model_points:
        raise ValueError("large_model_points must not be empty")

    small_color: str = "#009688"
    large_color: str = "#ff5a45"

    fig, left_axis = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    right_axis = left_axis.twinx()

    small_cpu_counts: list[int] = [point.cpu_count for point in small_model_points]
    small_seconds: list[float] = [
        point.average_timestep_seconds for point in small_model_points
    ]
    large_cpu_counts: list[int] = [point.cpu_count for point in large_model_points]
    large_seconds: list[float] = [
        point.average_timestep_seconds for point in large_model_points
    ]

    small_line = left_axis.plot(
        small_cpu_counts,
        small_seconds,
        marker="o",
        linewidth=2.5,
        markersize=7,
        color=small_color,
        label=(
            f"{small_model_points[0].region_name} "
            f"({small_model_points[0].basin_area_km2:,.0f} km²)"
        ),
    )
    large_line = right_axis.plot(
        large_cpu_counts,
        large_seconds,
        marker="o",
        linewidth=2.5,
        markersize=7,
        color=large_color,
        label=(
            f"{large_model_points[0].region_name} "
            f"({large_model_points[0].basin_area_km2:,.0f} km²)"
        ),
    )

    all_cpu_counts: list[int] = sorted(set(small_cpu_counts + large_cpu_counts))
    left_axis.set_xticks(all_cpu_counts)
    left_axis.set_xlabel("Number of CPUs", fontsize=14, fontweight="bold")
    left_axis.set_ylabel(
        "Time small model (seconds)",
        fontsize=13,
        fontweight="bold",
        color=small_color,
    )
    right_axis.set_ylabel(
        "Time large model (seconds)",
        fontsize=13,
        fontweight="bold",
        color=large_color,
    )
    left_axis.tick_params(axis="y", colors=small_color)
    right_axis.tick_params(axis="y", colors=large_color)

    left_axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
    left_axis.set_title(
        "Large-scale spinup runtime scaling",
        fontsize=15,
        fontweight="bold",
    )

    lines = small_line + large_line
    labels: list[str] = [line.get_label() for line in lines]
    left_axis.legend(lines, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot runtime scaling from large-scale GEB spinup logs. The runtime "
            "value is the average of the final timestep timings in each log."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing region folders. Default: {DEFAULT_MODELS_DIR}",
    )
    parser.add_argument(
        "--small-region",
        default="Europe_004",
        help="Region to plot on the left y-axis as the small model.",
    )
    parser.add_argument(
        "--large-region",
        default="Europe_005",
        help="Region to plot on the right y-axis as the large model.",
    )
    parser.add_argument(
        "--timestep-count",
        type=int,
        default=2,
        help="Number of final timesteps to average.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output figure path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path. Default: {DEFAULT_CSV_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the runtime scaling plot workflow."""
    args = parse_args()
    small_model_points: list[RuntimePoint] = collect_region_points(
        models_dir=args.models_dir,
        region_name=args.small_region,
        timestep_count=args.timestep_count,
    )
    large_model_points: list[RuntimePoint] = collect_region_points(
        models_dir=args.models_dir,
        region_name=args.large_region,
        timestep_count=args.timestep_count,
    )
    all_points: list[RuntimePoint] = small_model_points + large_model_points

    write_runtime_csv(points=all_points, csv_path=args.csv)
    plot_runtime_scaling(
        small_model_points=small_model_points,
        large_model_points=large_model_points,
        output_path=args.output,
    )

    print(f"Wrote figure: {args.output}")
    print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
