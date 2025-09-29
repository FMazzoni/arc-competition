import numpy as np

from arc_competition.task_viewer import Grid


def calculate_difference_grid(
    current_grid: Grid,
    output_grid: Grid,
) -> Grid:
    """
    Generate difference grid comparing current_grid to output_grid.

    Args:
        current_grid: The current/solution grid (2D list of integers)
        output_grid: The expected output grid (2D list of integers)

    Returns:
        2D list of integers (Grid) where:
        - 0 = matched elements (current_grid[row][col] == output_grid[row][col])
        - 1 = unmatched elements (current_grid[row][col] != output_grid[row][col])

    Note:
        Always returns difference grid in the shape of the larger grid.
        Missing elements in smaller grid are treated as unmatched (padding with zeros).
    """
    current_arr = np.array(current_grid)
    output_arr = np.array(output_grid)

    # Create difference array in the shape of the larger grid
    diff_height = max(current_arr.shape[0], output_arr.shape[0])
    diff_width = max(current_arr.shape[1], output_arr.shape[1])
    difference_arr = np.ones((diff_height, diff_width), dtype=int)

    # Compare overlapping region
    current_rows, current_cols = current_arr.shape
    output_rows, output_cols = output_arr.shape
    min_rows = min(current_rows, output_rows)
    min_cols = min(current_cols, output_cols)

    if min_rows > 0 and min_cols > 0:
        # Compare the overlapping region
        current_overlap = current_arr[:min_rows, :min_cols]
        output_overlap = output_arr[:min_rows, :min_cols]
        matched_overlap = current_overlap == output_overlap
        difference_arr[:min_rows, :min_cols] = (~matched_overlap).astype(int)

    # Elements outside the overlapping region remain unmatched (1)

    return Grid(difference_arr.tolist())


def calculate_score(
    current_grid: Grid,
    output_grid: Grid,
    *,
    # Tunable parameters
    base_score: float = 100.0,
    size_mismatch_penalty: float = 50.0,
    unmatched_element_penalty: float = 1.0,
) -> float:
    """
    Calculate score based on grid comparison with size mismatch penalty.

    Args:
        current_grid: The current/solution grid (2D list of integers)
        output_grid: The expected output grid (2D list of integers)
        base_score: Starting score (default: 100.0)
        size_mismatch_penalty: Points deducted for size mismatch (default: 50.0)
        unmatched_element_penalty: Points deducted per unmatched element (default: 1.0)

    Returns:
        Score between -inf and 100 (or adjusted by base_score parameter)

    Rules Applied:
        - Start with base_score (default 100)
        - Deduct size_mismatch_penalty if current_grid and output_grid sizes differ (default 50)
        - Deduct unmatched_element_penalty for each unmatched element (default 1)
        - All comparison is done element-wise using top-left as reference point
    """
    score = base_score

    # Generate difference grid internally
    diff_grid = calculate_difference_grid(current_grid, output_grid)

    # Check for size mismatch
    current_arr = np.array(current_grid)
    output_arr = np.array(output_grid)
    if current_arr.shape != output_arr.shape:
        score -= size_mismatch_penalty

    # Convert difference grid to numpy array for efficient operations
    diff_arr = np.array(diff_grid)

    # Count unmatched elements (ones in the difference grid)
    unmatched_count = np.sum(diff_arr)

    score -= unmatched_count * unmatched_element_penalty

    return score
