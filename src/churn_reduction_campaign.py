import asyncio
from agent.agent_session import WorkflowContext, AgentSession
from agent.planner_agent import AGENT_NAME, planner_agent
from agents import set_tracing_disabled

"""
Task: Recommend churn reduction campaign.
"""


def print_simulation(simulation: dict):
    evaluation = simulation["evaluation"]
    print(f"\nSimulation: {simulation['action']}")
    print(f"Segment: {simulation['segment']['definition']}, {simulation['segment']['size']:,} subscriptions")
    print(f"Goal: churn -{evaluation['goal']['target_churn_reduction_pp']}pp, "
          f"revenue decline <= ${evaluation['goal']['max_revenue_decline']:,.0f} -> {evaluation['verdict']}")
    print(f"{'Discount':>8} {'Mean churn':>11} {'Change pp':>10} {'Revenue change':>15} {'Churn ok':>9} {'Revenue ok':>11}")
    for s in evaluation["scenarios"]:
        print(f"{s['discount_rate']:>8.0%} {s['segment_mean_churn_probability']:>11.4%} "
              f"{s['churn_change_pp']:>+10.4f} {s['revenue_change']:>+15,.2f} "
              f"{str(s['meets_churn_target']):>9} {str(s['meets_revenue_constraint']):>11}")


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
