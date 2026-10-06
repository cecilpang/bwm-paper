import json
from agents import Agent, ModelSettings
from openai.types.shared import Reasoning
from agents.tool import function_tool
from agent.bwm_db import get_feasible_actions
from agent.llm_connection import get_gemini


AGENT_NAME = "action_recommendation_agent"


@function_tool
async def get_action_space() -> str:
    """
    Get information of the available actions from the action space.
    """
    return json.dumps(get_feasible_actions(), indent=2)


action_recommendation_agent = Agent(
    name=AGENT_NAME,
    instructions=f"""
    Your role is to recommend actions based on the context given to you by the controller agent. 
    1. Use tool to retrive the feasible actions from the action space.
    2. Determine which of the actions would be useful.
    3. Output the selected actions and the reason they are useful.
    Do not suggest any actions other than those in the action space. 
    If the action space is not available, return the error message and stop.
    """,
    tools=[get_action_space],
    model='gpt-6.1-sol',
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
)

