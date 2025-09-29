"""
Agent schemas for the ARC Competition
"""

from typing import Any, Literal

from atomic_agents.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field

# Import Grid type from task_viewer
from .task_viewer import Grid


# Grid Analysis Schemas
class GridAnalysisInput(BaseIOSchema):
    """Input schema for ARC grid analysis."""

    input_grid: Grid = Field(description="Input grid as 2D list of integers")
    output_grid: Grid = Field(description="Output grid as 2D list of integers")


class GridAnalysisOutput(BaseIOSchema):
    """Output schema for ARC grid analysis."""

    # Overall summary
    analysis_summary: str = Field(description="Overall summary of the analysis")

    # Detected patterns
    detected_patterns: list[str] = Field(
        description="Patterns detected across the example: copy "
    )

    # Confidence assessment
    overall_solution_confidence: float = Field(
        description="Overall confidence in understanding the pattern"
    )
