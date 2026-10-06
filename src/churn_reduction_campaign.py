import asyncio
from agent.agent_session import WorkflowContext, AgentSession
from agent.planner_agent import AGENT_NAME, planner_agent
from agents import set_tracing_disabled

"""
Task: Recommend churn reduction campaign.
"""


def print_simulation(simulation: dict):
    goal, objective, revenue = simulation["goal"], simulation["objective_kpi"], simulation["revenue_kpi"]
    evaluation = simulation["evaluation"]
    segment = simulation["segment"]
    print(f"\nSegment: {' and '.join(segment['filters']) or 'all'}: "
          f"{segment['size']:,} of {segment['population_size']:,} {simulation['unit_entity']} rows")
    print(f"Objective: {goal['direction']} {objective['kpi_name']} ({objective['scope']}) "
          f"by >= {goal['target_change']:g} {goal['target_mode']}")
    if revenue:
        print(f"Revenue constraint: {revenue['kpi_name']} decline <= {goal['max_revenue_decline']:,.0f}")
    print(f"Verdict: {evaluation['verdict']}; recommended: "
          f"{(evaluation['recommended'] or evaluation['best_within_budget'] or {}).get('label')}")
    width = max(len(s["label"]) for s in simulation["scenarios"])
    print(f"{'Scenario':<{width}} {objective['kpi_name']:>24} {'Improvement':>12} "
          f"{'Revenue change':>15} {'Target ok':>9} {'Revenue ok':>10}")
    for s in simulation["scenarios"]:
        print(f"{s['label']:<{width}} {s['objective_value']:>24.6g} {s['objective_improvement']:>+12.4f} "
              f"{s['revenue_change']:>+15,.2f} {str(s['meets_objective_target']):>9} "
              f"{str(s['meets_revenue_constraint']):>10}")


async def main():
        # Do not send tracing data to OpenAI
        set_tracing_disabled(True)

        goal = """
            Reduce the churn rate of high frequency subscripions (top 10%) by at least 0.1 percentage point.
            For example from 2.0% to 1.9%.
            The total monthly revuenue must not decline by more than $200,000
        """

        context = WorkflowContext()
        agent_session = AgentSession(AGENT_NAME, context, planner_agent)
        r = await agent_session.run(goal)
        print(r.model_dump_json(indent=2))

        if "simulation" in context.state:
            print_simulation(context.state["simulation"])

if __name__ == "__main__":
    asyncio.run(main())
