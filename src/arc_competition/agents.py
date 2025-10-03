"""
Analysis Agents for the ARC Competition
Including data analysis and grid pattern analysis agents
"""

import json
import uuid
from datetime import datetime

from atomic_agents import AgentConfig, AtomicAgent
from atomic_agents.base.base_io_schema import BaseIOSchema
from atomic_agents.context import SystemPromptGenerator
from atomic_agents.context.chat_history import ChatHistory
from e2b_code_interpreter import Sandbox
from loguru import logger
from pydantic import Field

from .config import config

# Import Grid type from task_viewer
from .task_viewer import ARCChallenge, Grid, GridExample
from .utils import calculate_difference_grid, calculate_score


class ARCChallengeInput(BaseIOSchema, ARCChallenge):
    """Grid input to the ARC Challenge"""

    ...


class GridExampleExtended(GridExample):
    """Extends schema for GridExample."""

    current: Grid = Field(description="Current grid as 2D list of integers")
    difference: Grid = Field(description="Difference grid as 2D list of 1s and 0s")
    score: float = Field(description="Score of the current grid")


# Grid Analysis Schemas
class FunctionAgentInput(BaseIOSchema):
    """Input schema for ARC grid analysis."""

    example_grids: list[GridExampleExtended] = Field(
        description="Example grids containing the input and output grids"
    )

    def calculate_total_score(self) -> float:
        return sum([x.score for x in self.example_grids])


class FunctionAgentOutput(BaseIOSchema):
    """Output schema for ARC grid analysis."""

    # Overall summary
    analysis_summary: str = Field(description="Overall summary of the analysis")

    function_to_execute: str = Field(
        description="python function to execute, the function is called transform grid, which takes in a list[list[int]]"
    )


class ChallengeSolverAgentOutput(BaseIOSchema):
    """Output schema for ARC grid analysis."""

    # Overall summary
    functions_ran: list[str] = Field(description="Functions ran")

    final_grid: Grid = Field(description="Final grid")


def create_function_agent(
    **kwargs,
) -> AtomicAgent[FunctionAgentInput, FunctionAgentOutput]:
    """Create a grid pattern analysis agent specialized for ARC challenges."""

    # Create optimized system prompt generator for grid analysis
    system_prompt_generator = SystemPromptGenerator(
        background=[
            "You are an expert in pattern recognition and visual reasoning, specialized in analyzing grid-based puzzles.",
            "You have deep expertise in ARC (Abstraction and Reasoning Corpus) challenges and can identify subtle patterns.",
            "You understand both low-level pixel patterns and high-level conceptual transformations.",
            "Given an input grid, current grid and an output grid, you can identify the next best function to execute to transform the current grid into the output grid.",
            "Only use small python functions, preferably less than 10 lines of code but no more than 50 lines of code.",
            "You will also be given a score for the current grid, you can use this to identify issues with the functions or the order of execution.",
            "The score is a float between 0 and 100, where 0 is the worst and 100 is the best.",
            "50 points are deducted for mismatch output sizes between the current grid and the output grid.",
            "1 point is deducted for each element in the current grid that is unmatched in the output grid.",
            "all comparisons will be done with the top left element of the current grid as the reference point.",
            "a reference difference grid will be provided to you, you can use this to identify the differences between the current grid and the output grid.",
            "The reference difference grid will be a 2D grid of 0s and 1s, where 0s are the elements that are matched and 1s are the elements that are unmatched.",
        ]
    )

    # Create the grid analysis agent
    function_agent = AtomicAgent[FunctionAgentInput, FunctionAgentOutput](
        config=AgentConfig(
            client=config.client,
            model=config.model.name,
            system_prompt_generator=system_prompt_generator,
            model_api_parameters=config.model.get_api_parameters(),
            **kwargs,
        )
    )

    return function_agent


def update_to_extended_input(
    input: Grid, current: Grid, output: Grid
) -> GridExampleExtended:
    return GridExampleExtended(
        input=input,
        output=output,
        current=current,
        score=calculate_score(current, output),
        difference=calculate_difference_grid(current, output),
    )


def run_code_offline(code: str, input: Grid) -> Grid:
    code = code + "\n" + f"result = transform_grid({input})"

    # Create a namespace to capture variables
    namespace = {}
    exec(code, namespace)

    # Access the result
    result = namespace["result"]
    result = Grid(result)
    return result


def run_code_e2b(code: str, input: Grid) -> Grid:
    sbx = Sandbox.create()  # By default the sandbox is alive for 5 minutes
    execution = sbx.run_code(
        code + "\n" + f"transform_grid({input})"
    )  # Execute Python inside the sandbox
    output = execution.results[0].json
    if not isinstance(output, list):
        raise ValueError(f"Output is not a list: {output}")
    return Grid(output)


def run_code(code: str, input: Grid) -> Grid:
    if config.api.e2b_api_key:
        return run_code_e2b(code, input)
    else:
        return run_code_offline(code, input)


def update_example_grids(
    example_grids: list[GridExampleExtended], function_to_execute: str
) -> list[GridExampleExtended]:
    new_example_grids = []
    for example in example_grids:
        updated_example = example.model_copy()
        updated_example.current = run_code(function_to_execute, updated_example.current)
        updated_example.score = calculate_score(
            updated_example.current, updated_example.output
        )
        updated_example.difference = calculate_difference_grid(
            updated_example.current, updated_example.output
        )
        new_example_grids.append(updated_example)
    return new_example_grids


def create_challenge_solver_agent() -> AtomicAgent[
    ARCChallengeInput, ChallengeSolverAgentOutput
]:
    print("Creating challenge solver agent")
    history = ChatHistory()
    wrapper_agent = AtomicAgent[ARCChallengeInput, ChallengeSolverAgentOutput](
        config=AgentConfig(
            client=config.client,
            model=config.model.name,
            model_api_parameters=config.model.get_api_parameters(),
            history=history,
        )
    )
    function_agent = create_function_agent(history=history)

    def wrapper_run(
        user_input: ARCChallengeInput | None = None,
    ) -> ChallengeSolverAgentOutput:
        """Wrapper function with comprehensive logging"""
        if user_input is None:
            raise ValueError("Input data is required")

        # Generate unique run ID and setup logging
        run_id = str(uuid.uuid4())[:8]

        # Simple file logging setup
        log_file = (
            f"logs/agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_run_{run_id}.log"
        )
        logger.add(log_file, level="DEBUG", rotation="10 MB", retention="30 days")

        # Log run start with all metadata
        run_start_data = {
            "event": "RUN_START",
            "run_id": run_id,
            "challenge_data": user_input.model_dump(),
        }
        logger.info(f"RUN_START|{json.dumps(run_start_data, default=str)}")

        examples = [
            update_to_extended_input(example.input, example.input, example.output)
            for example in user_input.train
        ]
        function_agent_input = FunctionAgentInput(
            example_grids=examples,
        )

        num_examples = len(user_input.train)
        num_loops = 0
        total_score = function_agent_input.calculate_total_score()
        functions_ran = []

        # Log initial state
        initial_state_data = {
            "event": "INITIAL_STATE",
            "run_id": run_id,
            "num_examples": num_examples,
            "total_score": total_score,
            "function_agent_input": function_agent_input.model_dump(),
        }
        logger.info(f"INITIAL_STATE|{json.dumps(initial_state_data, default=str)}")

        while total_score < 100 * num_examples and num_loops < 5:
            iteration = num_loops + 1

            # Log iteration input
            iteration_input_data = {
                "event": "ITERATION_INPUT",
                "run_id": run_id,
                "iteration": iteration,
                "input": function_agent_input.model_dump(),
            }
            logger.info(
                f"ITERATION_INPUT|{json.dumps(iteration_input_data, default=str)}"
            )

            output = function_agent.run(function_agent_input)

            # Log iteration output
            iteration_output_data = {
                "event": "ITERATION_OUTPUT",
                "run_id": run_id,
                "iteration": iteration,
                "output": output.model_dump(),
            }
            logger.info(
                f"ITERATION_OUTPUT|{json.dumps(iteration_output_data, default=str)}"
            )

            function_agent_input.example_grids = update_example_grids(
                function_agent_input.example_grids, output.function_to_execute
            )
            functions_ran.append(output.function_to_execute)

            total_score = function_agent_input.calculate_total_score()
            num_loops += 1

            # Log iteration completion
            iteration_complete_data = {
                "event": "ITERATION_COMPLETE",
                "run_id": run_id,
                "iteration": iteration,
                "total_score": total_score,
                "function_executed": output.function_to_execute,
            }
            logger.info(
                f"ITERATION_COMPLETE|{json.dumps(iteration_complete_data, default=str)}"
            )

        # Log run end
        run_end_data = {
            "event": "RUN_END",
            "run_id": run_id,
            "total_loops": num_loops,
            "final_score": total_score,
            "functions_executed": functions_ran,
            "success": total_score >= 100 * num_examples,
        }
        logger.info(f"RUN_END|{json.dumps(run_end_data, default=str)}")

        return ChallengeSolverAgentOutput(
            functions_ran=functions_ran,
            final_grid=function_agent_input.example_grids[0].current,
        )

    wrapper_agent.run = wrapper_run

    return wrapper_agent
