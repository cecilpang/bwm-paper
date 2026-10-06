from typing import List, Literal
from pydantic import BaseModel, Field
from agents import Agent, ModelSettings
from openai.types.shared import Reasoning
from agent.llm_connection import get_gemini
from agent.ml_model_discovery_agent import ml_model_discovery_agent
from agent.action_recommendation_agent import action_recommendation_agent
from agent.data_query_agent import data_query_agent
from agent.simulation_agent import simulation_agent


AGENT_NAME = "planner_agent"


class ScenarioResult(BaseModel):
    action: str = Field(description="The simulated action, e.g. '10% discount on monthly_cost'")
    segment_mean_churn_probability: float
    churn_change_pp: float = Field(description="Change of the segment mean churn probability, in percentage points")
    revenue_change: float = Field(description="Change of total expected monthly revenue, in dollars")
    meets_churn_target: bool
    meets_revenue_constraint: bool


class CampaignRecommendation(BaseModel):
    goal_interpretation: str = Field(description="Segment, churn metric, target and revenue constraint used")
    models_used: List[str]
    baseline: str = Field(description="Baseline segment size, mean churn probability and expected monthly revenue")
    scenarios: List[ScenarioResult]
    verdict: Literal["feasible", "infeasible"]
    recommended_action: str
    tradeoff_options: List[str] = Field(description="Alternatives when the goal cannot be fully met")
    caveats: List[str]


planner_agent = Agent(
    name=AGENT_NAME,
    instructions=f"""
    Your role is to recommend a campaign that achieves the goal in the user prompt.
    1. Use agent as tool to get info of useful machine learning (ML) models from the model registry.
    2. Based on the result of step 1, use agent as tool to obtain feasible actions.
    3. Use agent as tool to query the database for the baseline of the population targeted by the goal
       (segment size, percentile cutoff, mean churn probability, monthly revenue) and for business rules.
    4. Translate the goal into a simulation spec (segment attribute and top share of subscriptions, churn reduction
       target in absolute percentage points, revenue-decline cap in dollars) and use agent as tool to
       run the simulations of the feasible actions and evaluate them against the goal.
    5. Based on the results of steps 1-4, recommend a campaign:
       - If a scenario meets the goal, recommend the feasible scenario with the smallest revenue loss.
       - If none does, set the verdict to infeasible, recommend the best scenario within the revenue
         budget, and list as trade-off options the cheapest scenario that meets the churn target
         (or state that none does) and the scenario with the largest churn reduction and its cost.
       - Mention any business rules that constrain the campaign.
    If step 1 or 2 returns no results, do not recommend anything and stop.
    Do not suggest any actions other than the feasible actions. Use only numbers returned by the tools.
    Note in the caveats that the churn model was trained on observational data, so simulated
    effects are model sensitivities and not causal estimates.
    """,
    tools=[ml_model_discovery_agent.as_tool(
            tool_name="ml_model_discovery_agent",
            tool_description="Find Machine Learning models in the model registry that are useful based on the user prompt.",
        ),
        action_recommendation_agent.as_tool(
            tool_name="action_recommendation_agent",
            tool_description="Recommend feasible actions from the action space based on the user prompt."
        ),
        data_query_agent.as_tool(
            tool_name="data_query_agent",
            tool_description="Answer data questions by running read-only queries against the database, and report business rules."
        ),
        simulation_agent.as_tool(
            tool_name="simulation_agent",
            tool_description=(
                "Simulate feasible actions with the churn prediction model and evaluate them against the goal. "
                "Input must state the segment (attribute and top share, e.g. top 10% by frequency), the churn reduction target in "
                "absolute percentage points and the revenue-decline cap in dollars."
            ),
        )],
    output_type=CampaignRecommendation,
    model='gpt-6.1-sol',
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
)
