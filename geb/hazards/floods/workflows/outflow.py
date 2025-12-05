"""Workflow to detect outflow points in flood masks."""

import numpy as np

from geb.types import TwoDArrayBool, TwoDArrayInt64


def trace_border_4_connectivity(
    mask: TwoDArrayBool,
    row: int,
    col: int,
    n_cells: int,
    counter_clockwise: bool = False,
) -> TwoDArrayInt64:
    """
    Traces the boundary N steps in the specified direction using 4-connectivity.

    This uses the Moore-Neighbor Tracing algorithm specialized for 4-connectivity,
    allowing only North, East, South, and West movements.

    Args:
        mask: 2D boolean array (True is inside).
        row: Row coordinate of the start point.
        col: Column coordinate of the start point.
        n_cells: Number of steps to trace.
        counter_clockwise: If True, traces counter-clockwise. Defaults to False.

    Returns:
        An array of (r, c) coordinates forming the traced path,
                    starting from the start_point.
    """
    rows, cols = mask.shape

    # Moore Neighborhood offsets (4-connectivity)
    if counter_clockwise:
        # Starting from North, Counter-Clockwise: (N, W, S, E)
        neighbors = np.array([(-1, 0), (0, -1), (1, 0), (0, 1)])
    else:
        # Starting from North, Clockwise: (N, E, S, W)
        neighbors = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])

    path = [(row, col)]

    # Find the initial 'backtrack' direction (the first adjacent False pixel)
    # The search will start immediately clockwise from this background direction.
    backtrack_idx = -1
    for i in range(4):
        dr, dc = neighbors[i]
        nr, nc = row + dr, col + dc

        # Check boundary and mask value (False means background/out-of-bounds)
        if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr, nc]:
            backtrack_idx = i
            break

    if backtrack_idx == -1:
        # Start point is fully surrounded by True, not a border point
        return np.array(path)

    # Tracing Loop
    curr_r, curr_c = row, col

    for step in range(n_cells):
        found_next = False

        # Start searching from the position immediately after the backtrack pixel (to hug the wall)
        for k in range(4):
            # idx is the index of the current neighbor being checked
            idx = (backtrack_idx + 1 + k) % 4

            dr, dc = neighbors[idx]
            nr, nc = curr_r + dr, curr_c + dc

            # Check if the potential next cell is valid (on map AND True in mask)
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc]:
                # Found the next boundary pixel!

                # Update current position and path
                curr_r, curr_c = nr, nc
                path.append((nr, nc))

                # Update the backtrack reference for the next iteration:
                # The new backtrack index is 180 degrees opposite of the move just made.
                backtrack_idx = (idx + 2) % 4

                found_next = True
                break

        if not found_next:
            # Reached a dead end
            break

    # Convert the final list of tuples to a NumPy array for clean output
    return np.array(path)


def create_outflow_in_mask(
    mask: TwoDArrayBool, row: int, col: int, width_cells: int
) -> TwoDArrayBool:
    """Create the outflow in the binary mask.

    The outflow is centered on the specified cell and extends
    outward along the border of the flood mask.

    Args:
        mask: 2D boolean array where True indicates presence of water.
        row: Row index of the cell to check for outflow.
        col: Column index of the cell to check for outflow.
        width_cells: Width of the outflow in number of cells.
            Must be an odd positive integer.

    Returns:
        A 2D boolean array indicating outflow points.

    Raises:
        ValueError: If width_cells is not an odd positive integer.
        ValueError: If the specified cell is not part of the flood mask.
        ValueError: If the specified cell is not part of the border of the flood mask.
    """
    if width_cells % 2 == 0 or width_cells < 1:
        raise ValueError("width_cells must be an odd positive integer.")

    if not mask[row, col]:
        raise ValueError("The specified cell is not part of the flood mask.")

    # Check if the cell is on the border of the flood mask
    rows, cols = mask.shape
    is_border = False
    for dr, dc in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr, nc]:
            is_border = True
            break

    if not is_border:
        raise ValueError("The specified cell is not on the border of the flood mask.")

    outflows = np.zeros_like(mask, dtype=bool)
    if width_cells == 1:
        # Single cell width, outflow is just the cell itself
        outflows[row, col] = True
    else:
        steps = (width_cells - 1) // 2
        # Trace in both directions
        contour_cw = trace_border_4_connectivity(
            mask, row, col, steps, counter_clockwise=False
        )
        contour_ccw = trace_border_4_connectivity(
            mask, row, col, steps, counter_clockwise=True
        )
        for contour_row, contour_col in contour_cw:
            outflows[contour_row, contour_col] = True
        for contour_row, contour_col in contour_ccw:
            outflows[contour_row, contour_col] = True

    return outflows
