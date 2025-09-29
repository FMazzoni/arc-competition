"""
ARC Competition Task Viewer

This module provides functionality to load and visualize ARC (Abstraction and Reasoning Corpus)
challenges in a nicely formatted HTML view suitable for Jupyter notebooks.
"""

import json
from pathlib import Path
from typing import NewType

from IPython.display import HTML, display
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).parent.parent.parent

# A 2D grid representing an ARC puzzle grid where each cell contains an integer
# representing a color or object type. Grids are typically small (e.g., 3x3 to 30x30)
# and are used for both input and output examples in ARC challenges.
Grid = NewType("Grid", list[list[int]])


def serialize_grid(grid: Grid) -> str:
    """
    Serialize a 2D grid to clean text representation.
    """
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


class GridExample(BaseModel):
    """Represents a single input/output example in an ARC challenge."""

    input: Grid = Field(description="Input grid as 2D list of integers")
    output: Grid = Field(description="Output grid as 2D list of integers")

    def prompt_format(self) -> str:
        """
        Format the example for LLM prompt consumption.

        Returns:
            Formatted string with input and output grids for prompt use
        """
        input_text = serialize_grid(self.input)
        output_text = serialize_grid(self.output)

        return f"Input:\n{input_text}\n\nOutput:\n{output_text}"


class ARCChallenge(BaseModel):
    """Represents a complete ARC challenge with training and test examples."""

    train: list[GridExample] = Field(description="Training examples")
    test: list[GridExample] = Field(description="Test examples (input only)")


class SubmissionAttempt(BaseModel):
    """Represents a submission attempt for a challenge."""

    attempt_1: Grid = Field(description="First attempt output grid")
    attempt_2: Grid = Field(description="Second attempt output grid")


class ARCDataLoader:
    """General data loader for ARC challenge files."""

    @staticmethod
    def load_challenges(file_path: str | Path) -> dict[str, ARCChallenge]:
        """
        Load ARC challenges from a JSON file.

        Args:
            file_path: Path to the challenges JSON file

        Returns:
            Dictionary mapping challenge IDs to ARCChallenge objects
        """
        file_path = Path(file_path)
        with file_path.open("r") as f:
            raw_data = json.load(f)

        challenges = {}
        for challenge_id, challenge_data in raw_data.items():
            # Convert training examples
            train_examples = [
                GridExample(input=ex["input"], output=ex["output"])
                for ex in challenge_data["train"]
            ]

            # Convert test examples (input only)
            test_examples = [
                GridExample(
                    input=ex["input"], output=Grid([])
                )  # Test examples don't have outputs
                for ex in challenge_data["test"]
            ]

            challenges[challenge_id] = ARCChallenge(
                train=train_examples, test=test_examples
            )

        return challenges

    @staticmethod
    def load_challenge(file_path: str | Path, challenge_id: str) -> ARCChallenge:
        """
        Load an ARC challenge from a JSON file.

        Args:
            file_path: Path to the challenges JSON file
            challenge_id: ID of the challenge to load

        Returns:
            ARCChallenge object
        """
        file_path = Path(file_path)
        challenges = ARCDataLoader.load_challenges(file_path)
        return challenges[challenge_id]

    @staticmethod
    def load_solutions(file_path: str | Path) -> dict[str, list[SubmissionAttempt]]:
        """
        Load ARC solutions from a JSON file.

        Args:
            file_path: Path to the solutions JSON file

        Returns:
            Dictionary mapping challenge IDs to lists of submission attempts
        """
        file_path = Path(file_path)
        with file_path.open("r") as f:
            raw_data = json.load(f)

        solutions = {}
        for challenge_id, attempts_list in raw_data.items():
            attempts = []
            for attempt_data in attempts_list:
                attempts.append(
                    SubmissionAttempt(
                        attempt_1=attempt_data["attempt_1"],
                        attempt_2=attempt_data["attempt_2"],
                    )
                )
            solutions[challenge_id] = attempts

        return solutions


def grid_to_html(grid: Grid, title: str = "", cell_size: int = 30) -> str:
    """
    Convert a 2D grid to HTML table representation.

    Args:
        grid: 2D list of integers representing the grid
        title: Optional title for the grid
        cell_size: Size of each cell in pixels

    Returns:
        HTML string representation of the grid
    """
    if not grid:
        return f"<div><strong>{title}</strong><br>Empty grid</div>"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    html = "<div style='margin: 10px;'>"
    if title:
        html += f"<strong>{title}</strong><br>"

    html += "<table style='border-collapse: collapse; margin: 5px;'>"

    for row in grid:
        html += "<tr>"
        for cell in row:
            # Create a color based on the cell value
            # Use a simple hash to generate consistent colors
            color_hash = hash(str(cell)) % 360
            bg_color = f"hsl({color_hash}, 70%, 80%)"
            border_color = f"hsl({color_hash}, 70%, 50%)"

            html += f"""
            <td style='
                width: {cell_size}px; 
                height: {cell_size}px; 
                border: 1px solid {border_color}; 
                background-color: {bg_color}; 
                text-align: center; 
                font-size: 12px;
                font-weight: bold;
            '>
                {cell}
            </td>
            """
        html += "</tr>"

    html += "</table>"
    html += f"<small>Size: {rows}×{cols}</small>"
    html += "</div>"

    return html


def task_viewer(file_path: str | Path, challenge_id: str) -> HTML:
    """
    Create a nicely formatted HTML view of an ARC task.

    Args:
        file_path: Path to the challenges JSON file
        challenge_id: ID of the challenge to display

    Returns:
        HTML object that can be displayed in a Jupyter notebook
    """
    # Load the challenges
    challenges = ARCDataLoader.load_challenges(file_path)

    if challenge_id not in challenges:
        available_ids = list(challenges.keys())[:10]  # Show first 10 IDs
        return HTML(f"""
        <div style='color: red; padding: 20px; border: 2px solid red; border-radius: 5px;'>
            <h3>Challenge not found!</h3>
            <p>Challenge ID '{challenge_id}' not found in the file.</p>
            <p>Available challenge IDs (first 10): {", ".join(available_ids)}</p>
        </div>
        """)

    challenge = challenges[challenge_id]

    # Start building the HTML
    html = f"""
    <div style='
        font-family: Arial, sans-serif; 
        max-width: 1200px; 
        margin: 20px auto; 
        padding: 20px;
        border: 2px solid #333;
        border-radius: 10px;
        background-color: #f9f9f9;
    '>
        <h2 style='text-align: center; color: #333; margin-bottom: 30px;'>
            ARC Challenge: {challenge_id}
        </h2>
    """

    # Training examples section
    html += """
    <div style='margin-bottom: 30px;'>
        <h3 style='color: #2c5aa0; border-bottom: 2px solid #2c5aa0; padding-bottom: 5px;'>
            Training Examples
        </h3>
    """

    for i, example in enumerate(challenge.train, 1):
        html += f"""
        <div style='
            margin: 20px 0; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            background-color: white;
        '>
            <h4 style='color: #666; margin-top: 0;'>Example {i}</h4>
            <div style='display: flex; gap: 20px; align-items: flex-start;'>
                {grid_to_html(example.input, "Input", cell_size=25)}
                <div style='font-size: 24px; color: #666; margin-top: 30px;'>→</div>
                {grid_to_html(example.output, "Output", cell_size=25)}
            </div>
        </div>
        """

    html += "</div>"

    # Test examples section
    html += """
    <div style='margin-bottom: 30px;'>
        <h3 style='color: #d63384; border-bottom: 2px solid #d63384; padding-bottom: 5px;'>
            Test Examples
        </h3>
    """

    for i, example in enumerate(challenge.test, 1):
        html += f"""
        <div style='
            margin: 20px 0; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            background-color: #fff8e1;
        '>
            <h4 style='color: #666; margin-top: 0;'>Test {i}</h4>
            <div style='display: flex; gap: 20px; align-items: flex-start;'>
                {grid_to_html(example.input, "Input", cell_size=25)}
                <div style='
                    width: 200px; 
                    height: 100px; 
                    border: 2px dashed #d63384; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    color: #d63384;
                    font-weight: bold;
                '>
                    Your Solution
                </div>
            </div>
        </div>
        """

    html += "</div>"

    # Summary section
    html += f"""
    <div style='
        margin-top: 30px; 
        padding: 15px; 
        background-color: #e8f4f8; 
        border-radius: 8px;
        border-left: 4px solid #2c5aa0;
    '>
        <h4 style='margin-top: 0; color: #2c5aa0;'>Challenge Summary</h4>
        <ul style='margin: 0;'>
            <li><strong>Training examples:</strong> {len(challenge.train)}</li>
            <li><strong>Test examples:</strong> {len(challenge.test)}</li>
            <li><strong>Challenge ID:</strong> {challenge_id}</li>
        </ul>
    </div>
    """

    html += "</div>"

    return HTML(html)


def display_challenge(file_path: str | Path, challenge_id: str) -> None:
    """
    Display an ARC challenge in a Jupyter notebook.

    Args:
        file_path: Path to the challenges JSON file
        challenge_id: ID of the challenge to display
    """
    html_view = task_viewer(file_path, challenge_id)
    display(html_view)


# Convenience functions for common file types
def view_training_challenge(
    challenge_id: str, data_dir: str | Path = "arc-prize-2025"
) -> None:
    """View a training challenge from the default data directory."""
    file_path = ROOT_DIR / data_dir / "arc-agi_training_challenges.json"
    display_challenge(file_path, challenge_id)


def view_test_challenge(
    challenge_id: str, data_dir: str | Path = "arc-prize-2025"
) -> None:
    """View a test challenge from the default data directory."""
    file_path = ROOT_DIR / data_dir / "arc-agi_test_challenges.json"
    display_challenge(file_path, challenge_id)


def view_evaluation_challenge(
    challenge_id: str, data_dir: str | Path = "arc-prize-2025"
) -> None:
    """View an evaluation challenge from the default data directory."""
    file_path = ROOT_DIR / data_dir / "arc-agi_evaluation_challenges.json"
    display_challenge(file_path, challenge_id)
