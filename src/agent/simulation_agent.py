import json
from agents import Agent, ModelSettings, RunContextWrapper
from openai.types.shared import Reasoning
from agents.tool import function_tool
from agent.tool_errors import tool_error_message
from agent.agent_session import WorkflowContext
from agent.simulation import allowed_discount_rates, evaluate_goal, simulate_uniform_discount


AGENT_NAME = "simulation_agent"


@function_tool(failure_error_function=tool_error_message)
async def get_simulation_actions() -> str:
    """
    Get the discount rates allowed by the feasible action on subscription.monthly_cost.
    """
    return json.dumps({"discount_rates": allowed_discount_rates()})


@function_tool(failure_error_function=tool_error_message)
async def run_discount_simulation(
    ctx: RunContextWrapper[WorkflowContext],
    segment_attribute: str,
    top_share: float,
    target_churn_reduction_pp: float,
    max_revenue_decline: float,
) -> str:
    """
    Simulate a uniform monthly_cost discount (every allowed rate, plus the 0% baseline) for the
    segment of subscriptions with the highest values of an attribute, using the churn
    prediction model, and evaluate each scenario against the goal.

    Args:
        segment_attribute: Attribute defining the segment: "frequency", "breadth", "tenure_days"
            or "monthly_cost".
        top_share: Share of subscriptions with the highest attribute values, as a fraction
            between 0 and 1, e.g. 0.10 for the top 10%.
        target_churn_reduction_pp: Required drop of the segment's mean churn probability, in
            absolute percentage points (2.0 means 2 pp).
        max_revenue_decline: Largest allowed decline of total expected monthly revenue, in dollars.
    """
    simulation = simulate_uniform_discount(segment_attribute, top_share)
    evaluation = evaluate_goal(simulation, target_churn_reduction_pp, max_revenue_decline)
    result = {**{k: v for k, v in simulation.items() if k != "scenarios"}, "evaluation": evaluation}
    ctx.context.state["simulation"] = result
    return json.dumps(result, indent=2)


simulation_agent = Agent(
    name=AGENT_NAME,
    instructions=f"""
    Your role is to run what-if simulations of feasible actions with the churn prediction model
    and evaluate them against the goal given by the controller agent.
    1. Use tool to retrieve the allowed discount rates.
    2. Translate the goal into the simulation parameters: the segment attribute and top share,
       the churn reduction target in absolute percentage points, and the revenue-decline cap in dollars.
    3. Use tool to run the simulation with those parameters.
    4. Output the parameters used, the verdict, the full scenario table (discount rate, segment mean
       churn probability, churn change in pp, revenue change, whether each constraint is met),
       the best scenario within the revenue budget, and the cheapest scenario that meets the churn target.
    Report numbers exactly as returned by the tool; do not compute or estimate any of your own.
    If a tool returns an error, correct the arguments according to the error message and retry;
    if it still fails, return the error message and stop.
    """,
    tools=[get_simulation_actions, run_discount_simulation],
    model='gpt-6.1-sol',
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
)
