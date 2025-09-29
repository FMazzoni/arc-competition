"""
Analysis Agents for the ARC Competition
Including data analysis and grid pattern analysis agents
"""

from atomic_agents import AgentConfig, AtomicAgent
from atomic_agents.context import SystemPromptGenerator

from .config import config
from .logging_config import get_logger
from .schemas import GridAnalysisInput, GridAnalysisOutput

logger = get_logger()


def create_grid_analysis_agent() -> AtomicAgent[GridAnalysisInput, GridAnalysisOutput]:
    """Create a grid pattern analysis agent specialized for ARC challenges."""

    client = config.client
    # Create optimized system prompt generator for grid analysis
    system_prompt_generator = SystemPromptGenerator(
        background=[
            "You are an expert in pattern recognition and visual reasoning, specialized in analyzing grid-based puzzles.",
            "You have deep expertise in ARC (Abstraction and Reasoning Corpus) challenges and can identify subtle patterns.",
            "You excel at recognizing transformations, symmetries, color progressions, spatial relationships, and object manipulation.",
            "You can break down complex visual transformations into simpler components and detect recurring patterns across examples.",
            "You understand both low-level pixel patterns and high-level conceptual transformations.",
            "Given an input grid and an output grid, you can identify the function that maps the input to the output.",
        ],
        steps=[
            "Examine each input-output pair carefully to understand the transformation.",
            "Synthesize findings into implementable solution strategies.",
        ],
        output_instructions=[
            "Provide a clear, comprehensive analysis of detected patterns.",
            "Explain transformations in terms that could be translated into programming logic.",
            "Give confidence scores for each detected pattern.",
            "Offer practical suggestions for implementing solutions.",
            "Highlight the most probable explanation if multiple interpretations exist.",
            "Include specific examples from the grids to support your analysis.",
            "Consider edge cases and potential complications in the pattern.",
        ],
    )

    # Create the grid analysis agent
    grid_agent = AtomicAgent[GridAnalysisInput, GridAnalysisOutput](
        config=AgentConfig(
            client=client,
            model=config.model.name,
            system_prompt_generator=system_prompt_generator,
            model_api_parameters=config.model.get_api_parameters(),
        )
    )

    return grid_agent
