import json
from typing import List, Optional
from agents import Agent, ModelSettings, RunContextWrapper
from openai.types.shared import Reasoning
from agents.tool import function_tool
from agent.tool_errors import tool_error_message
from agent.agent_session import WorkflowContext
from agent.bwm_db import get_attributes, get_feasible_actions, get_kpis, get_prediction_models
from agent.simulation import GoalSpec, SegmentFilter, simulate


AGENT_NAME = "simulation_agent"

# Largest number of scenarios returned to the model in full; the rest are summarised.
MAX_SCENARIOS_IN_RESPONSE = 60


@function_tool(failure_error_function=tool_error_message)
async def get_simulation_catalog() -> str:
    """
    Get what can be simulated: prediction models, feasible actions with their allowed levels,
    KPIs (named metrics a goal can target) and the attributes available for segment filters.
    """
    catalog = {
        "models": [
            {
                "model_id": m["model_id"], "short_desc": m["short_desc"],
                "unit_entity": m["unit_entity_name"],
                "prediction": f"{m['prediction_entity_name']}.{m['prediction_attribute_name']}",
                "prediction_kind": m["prediction_kind"],
                "features": [f"{f['entity_name']}.{f['attribute_name']}" for f in m["predictive_features"]],
            }
            for m in get_prediction_models()
        ],
        "actions": get_feasible_actions(),
        "kpis": get_kpis(),
        "attributes": [
            {"attribute": f"{a['entity_name']}.{a['attribute_name']}", "data_type": a["data_type"],
             "description": a["description"]}
            for a in get_attributes() if not a["is_key"]
        ],
    }
    return json.dumps(catalog, indent=2)


@function_tool(failure_error_function=tool_error_message)
async def run_simulation(
    ctx: RunContextWrapper[WorkflowContext],
    segment_filters: List[SegmentFilter],
    goal: GoalSpec,
    action_ids: Optional[List[int]],
    include_combinations: bool,
) -> str:
    """
    Simulate feasible actions applied uniformly to a segment, using the prediction models,
    and evaluate every scenario (baseline, each action level, and combinations across
    actions) against the goal.

    Args:
        segment_filters: Filters that together define the segment (all must hold).
        goal: The objective KPI target and the revenue constraint.
        action_ids: Feasible actions to simulate; null for all.
        include_combinations: Also simulate combinations of levels of different actions.
    """
    result = simulate(segment_filters, goal, action_ids, include_combinations)
    ctx.context.state["simulation"] = result

    response = dict(result)
    scenarios = result["scenarios"]
    if len(scenarios) > MAX_SCENARIOS_IN_RESPONSE:
        evaluation = result["evaluation"]
        highlighted = {0} | {
            pick["scenario_id"] for pick in evaluation.values() if isinstance(pick, dict)
        } | set(evaluation["feasible_scenario_ids"][:MAX_SCENARIOS_IN_RESPONSE])
        response["scenarios"] = [s for s in scenarios if s["scenario_id"] in highlighted]
        response["omitted_scenarios"] = len(scenarios) - len(response["scenarios"])
    return json.dumps(response, indent=2)


simulation_agent = Agent(
    name=AGENT_NAME,
    instructions=f"""
    Your role is to run what-if simulations of feasible actions with the prediction models
    and evaluate them against the goal given by the controller agent.
    1. Use tool to retrieve the simulation catalog: models, actions and levels, KPIs and attributes.
    2. Translate the goal into the simulation inputs, using only names from the catalog:
       - segment_filters: the population the goal targets, e.g. "top 10%" by an attribute is
         op "top_share" with value 0.10; thresholds use gt/ge/lt/le/eq/ne/between/in.
       - goal.objective_kpi: the KPI that best matches the goal's metric, its scope (segment or
         population), direction, target_change and target_mode. For fraction KPIs (unit
         probability/rate), "N percentage points" is percentage_points and "N%" of the baseline
         is relative_percent; state which reading you used.
       - goal.max_revenue_decline: the allowed revenue decline in currency units, or null.
       - action_ids: the actions relevant to the goal, or null for all.
    3. Use tool to run the simulation.
    4. Output the inputs used, the segment, the verdict, the scenario table (label, objective value,
       objective improvement, revenue change, whether each constraint is met), and the
       recommended, best-within-budget, cheapest-meeting-target and largest-improvement scenarios.
    Report numbers exactly as returned by the tool; do not compute or estimate any of your own.
    If a tool returns an error, correct the inputs according to the error message and retry;
    if it still fails, return the error message and stop.
    """,
    tools=[get_simulation_catalog, run_simulation],
    model='gpt-6.1-sol',
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
)
