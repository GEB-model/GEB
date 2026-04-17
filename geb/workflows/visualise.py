"""Functions for visualizing GEB model outputs."""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np


def plot_sunburst(
    hierarchy: dict[str, Any],
    title: str = "Water Circle",
    colors: dict[str, str] | None = None,
    figsize: tuple[int, int] = (10, 10),
    min_display_ratio: float = 0.02,
) -> plt.Figure:
    """Creates a sunburst plot using matplotlib to visualize a water circle hierarchy.

    Args:
        hierarchy: A dictionary representing the water flow hierarchy.
        title: Title of the plot.
        colors: Optional mapping of root sections to colors.
        figsize: Size of the figure.
        min_display_ratio: Minimum segment size as a fraction of the full circle.
            Segments smaller than this ratio are hidden.

    Returns:
        A matplotlib Figure object.

    Raises:
        ValueError: If min_display_ratio is outside the interval [0, 1).
    """
    if not 0 <= min_display_ratio < 1:
        raise ValueError("min_display_ratio must be in the interval [0, 1).")

    if colors is None:
        colors = {
            "in": "#636EFA",
            "out": "#EF5538",
            "storage change": "#D2D2D3",
            "balance": "#000000",
        }

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    # We want "in" and "out" to be balanced if possible, or just plot them as they are.
    # The original water_circle seems to sum up 'in' and 'out' and 'storage change'.

    def get_total_value(d: dict[str, Any] | float) -> float:
        if isinstance(d, (int, float)):
            return float(d)
        total = 0.0
        for k, v in d.items():
            if k == "_self":
                total += v
            else:
                total += get_total_value(v)
        return total

    total_value = get_total_value(hierarchy)
    if total_value == 0:
        return fig

    # Data structures to store plot segments
    # Each segment: (level, start_angle, width, label, color, value)
    segments: list[dict[str, Any]] = []

    def collect_segments(
        data: dict[str, Any],
        level: int = 0,
        start_angle: float = 0,
        total_span: float = 2 * np.pi,
        p_color: str | None = None,
        parent_value: float | None = None,
    ) -> None:
        current_angle = start_angle
        segment_total = get_total_value(data) if parent_value is None else parent_value

        # Preserve the order provided in the hierarchy.
        items = data.items()

        for name, value in items:
            if name == "_self":
                continue

            val = get_total_value(value)
            if val == 0:
                continue

            width = (val / segment_total) * total_span
            display_ratio = width / (2 * np.pi)

            if display_ratio < min_display_ratio:
                current_angle += width
                continue

            # Use root level's color as foundation if not in 'colors'
            seg_color = colors.get(name, p_color)

            segments.append(
                {
                    "level": level,
                    "start": current_angle,
                    "width": width,
                    "label": name,
                    "color": seg_color,
                    "value": val,
                }
            )

            if isinstance(value, dict):
                collect_segments(
                    value,
                    level + 1,
                    current_angle,
                    width,
                    seg_color,
                    val,
                )

            current_angle += width

    collect_segments(hierarchy)

    max_level = max(s["level"] for s in segments) if segments else 0
    inner_radius = 0.2
    outer_radius = 1.0
    level_width = (outer_radius - inner_radius) / (max_level + 1)

    for seg in segments:
        ax.bar(
            x=seg["start"] + seg["width"] / 2,
            height=level_width,
            width=seg["width"],
            bottom=inner_radius + seg["level"] * level_width,
            color=seg["color"],
            edgecolor="white",
            linewidth=0.5,
            align="center",
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # ty:ignore[unresolved-attribute]

    for seg in segments:
        level = seg["level"]
        start = seg["start"]
        width = seg["width"]
        label = seg["label"]
        mid_angle = start + width / 2

        r_inner = inner_radius + level * level_width
        r_outer = r_inner + level_width
        r_mid = r_inner + level_width / 2

        # Define styles
        fontsize: Literal[8, 9] = 8 if level > 0 else 9
        style: dict[str, str | int] = dict(
            fontweight="bold" if level == 0 else "normal",
            fontsize=fontsize,
            color="white" if level < 1 else "black",
            ha="center",
            va="center",
            clip_on=True,
        )

        def check_fit(
            text_obj: plt.Text, r1: float, r2: float, a1: float, a2: float
        ) -> bool:
            bbox = text_obj.get_window_extent(renderer)
            inv = ax.transData.inverted()
            # BBox corners in display -> data (theta, r)
            p0 = inv.transform((bbox.x0, bbox.y0))
            p1 = inv.transform((bbox.x1, bbox.y1))

            r_min, r_max = min(p0[1], p1[1]), max(p0[1], p1[1])
            # For angular width, we check the span of the bounding box
            # This is a bit simplified but works for standard rotations
            ang_span = abs(p1[0] - p0[0])

            # Use a slightly stricter margin (0.95) for safety
            return r_min >= r1 and r_max <= r2 and ang_span <= (a2 - a1) * 0.95

        placed = False

        # 1. Try Horizontal
        t = ax.text(mid_angle, r_mid, label, rotation=0, **style)  # ty:ignore[invalid-argument-type]
        if check_fit(t, r_inner, r_outer, start, start + width):
            placed = True
        else:
            t.remove()

        # 2. Try Tangential
        if not placed:
            rot = np.degrees(mid_angle) - 90
            if 90 < rot <= 270:
                rot -= 180
            t = ax.text(mid_angle, r_mid, label, rotation=rot, **style)  # ty:ignore[invalid-argument-type]
            if check_fit(t, r_inner, r_outer, start, start + width):
                placed = True
            else:
                t.remove()

        # 3. Try Radial
        if not placed:
            rot = np.degrees(mid_angle)
            if 90 < rot <= 270:
                rot -= 180
            t = ax.text(mid_angle, r_mid, label, rotation=rot, **style)  # ty:ignore[invalid-argument-type]
            if check_fit(t, r_inner, r_outer, start, start + width):
                placed = True
            else:
                t.remove()

        # 4. Final: Place outside with precise alignment
        if not placed:
            # Shift label a bit closer to the wedge's outer radius
            r_label = r_outer + 0.01
            rot_deg = np.degrees(mid_angle)

            # Normalize angle to [0, 360) for easier logic
            norm_angle = rot_deg % 360

            # Determine alignment based on which hemisphere the text is in
            # If on the right side (270° to 90°), ha="left" and rotation = angle
            # If on the left side (90° to 270°), ha="right" and rotation = angle - 180
            if 90 < norm_angle <= 270:
                actual_rot = norm_angle - 180
                ha = "right"
            else:
                actual_rot = (norm_angle + 180) % 360 - 180
                ha = "left"

            # Use more precise anchor for "outside" labels
            # ha set to 'left' means the start of text is at mid_angle, r_label
            # ha set to 'right' means the end of text is at mid_angle, r_label
            ax.text(
                mid_angle,
                r_label,
                label,
                rotation=actual_rot,
                ha=ha,
                va="center",
                fontsize=fontsize - 1,
                color="black",
                clip_on=False,
                # Using rotation_mode="anchor" ensures the 'ha' alignment
                # refers to the text box before it is rotated
                rotation_mode="anchor",
            )

    ax.set_axis_off()
    ax.set_title(title, pad=40, weight="bold", fontsize=14)
    return fig
