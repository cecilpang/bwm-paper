import json
from agents import Agent, ModelSettings
from openai.types.shared import Reasoning
from agents.tool import function_tool
from agent.autobus_db import get_prediction_models
from agent.llm_connection import get_gemini


AGENT_NAME = "ml_model_discovery_agent"


@function_tool
async def get_ml_models_info_from_registry() -> str:
    """
    Get information of the available ML models from the model registry.
    """
    return json.dumps(get_prediction_models(), indent=2)


ml_model_discovery_agent = Agent(
    name=AGENT_NAME,
    instructions=f"""
    Your role is to select machine learning (ML) models from the model registry 
    that can be used to plan actions that lead to the outcomes specified in the user prompt. 
    1. Use tool to retrive the names and descriptions of machine learning (ML) models from the model registry.
    2. Determine which of the ML models in the model registry would be useful.
    3. Output all the info found in the model registry.
    Do not suggest any models other than those in the registry. If the registry is not available, return the error message and stop.
    """,
    tools=[get_ml_models_info_from_registry],
    model='gpt-6.1-sol',
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
)

