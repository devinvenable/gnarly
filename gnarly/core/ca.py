"""Cellular automata rules and grid management."""

import random

import numpy as np


def initialize_grid(width: int, height: int) -> np.ndarray:
    """Initialize a grid with random 8x8 blocks.

    Args:
        width: Grid width in cells.
        height: Grid height in cells.

    Returns:
        2D numpy array with 0s and 1s.
    """
    grid = np.zeros((height, width), dtype=int)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if random.random() < 0.5:
                grid[y : y + 8, x : x + 8] = 1
    return grid


def update_grid(grid: np.ndarray, rule: str, divisor: int = 2) -> np.ndarray:
    """Update grid according to cellular automata rule.

    Args:
        grid: Current grid state.
        rule: Rule name ('Conway', 'HighLife', 'Seeds', 'Custom', 'DivisorRule').
        divisor: Divisor value for DivisorRule.

    Returns:
        Updated grid state.
    """
    neighbors = sum(
        np.roll(np.roll(grid, i, axis=0), j, axis=1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if not (i == 0 and j == 0)
    )

    if rule == "Conway":
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)
    elif rule == "HighLife":
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where(
            (grid == 0) & ((neighbors == 3) | (neighbors == 6)), 1, new_grid
        )
    elif rule == "Seeds":
        new_grid = np.zeros_like(grid)
        new_grid = np.where((grid == 0) & (neighbors == 2), 1, new_grid)
    elif rule == "Custom":
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 4)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)
    elif rule == "DivisorRule":
        remainder = neighbors % divisor
        new_value = remainder / divisor
        new_grid = np.where(new_value >= 0.5, 1, 0)
    else:
        # Default to Conway
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)

    return new_grid


class CAEngine:
    """Cellular automata engine for managing grid state."""

    RULES = ["Conway", "HighLife", "Seeds", "Custom", "DivisorRule"]

    def __init__(
        self, width: int, height: int, rule: str = "Conway", divisor: int = 2
    ):
        """Initialize the CA engine.

        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            rule: Initial rule name.
            divisor: Divisor value for DivisorRule.
        """
        self.width = width
        self.height = height
        self.rule = rule
        self.divisor = divisor
        self.grid = initialize_grid(width, height)

    def step(self) -> np.ndarray:
        """Advance the grid by one step.

        Returns:
            Updated grid state.
        """
        self.grid = update_grid(self.grid, self.rule, self.divisor)
        return self.grid

    def reset(self) -> np.ndarray:
        """Reset the grid to a new random state.

        Returns:
            New grid state.
        """
        self.grid = initialize_grid(self.width, self.height)
        return self.grid

    def is_dead(self) -> bool:
        """Check if the grid has no live cells.

        Returns:
            True if grid has no live cells.
        """
        return np.sum(self.grid) == 0
