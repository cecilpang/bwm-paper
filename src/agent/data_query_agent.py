import json
from agents import Agent, ModelSettings
from openai.types.shared import Reasoning
from agents.tool import function_tool
from agent.tool_errors import tool_error_message
from agent.autobus_db import get_business_rules, get_schema, run_select_query


AGENT_NAME = "data_query_agent"


@function_tool(failure_error_function=tool_error_message)
async def get_database_schema() -> str:
    """
    Get the tables, columns and column descriptions of the organization's data.
    """
    return json.dumps(get_schema(), indent=2)


@function_tool(failure_error_function=tool_error_message)
async def get_business_rules_info() -> str:
    """
    Get the business rules of the organization and the entities/attributes they govern.
    """
    return json.dumps(get_business_rules(), indent=2)


@function_tool(failure_error_function=tool_error_message)
async def query_database(sql: str) -> str:
    """
    Run a read-only SQLite query against the organization's data. At most 200 rows are returned.

    Args:
        sql: A single SELECT or WITH statement (SQLite dialect).
    """
    return json.dumps(run_select_query(sql), indent=2, default=str)


data_query_agent = Agent(
    name=AGENT_NAME,
    instructions=f"""
    Your role is to answer data questions from the controller agent by querying the database.
    1. Use tool to retrieve the database schema before writing any query.
    2. Use tool to retrieve the business rules.
    3. Write and run read-only SQLite queries that answer the question. Prefer aggregates over raw rows.
       SQLite has no PERCENTILE function; compute percentile cutoffs with ORDER BY ... LIMIT 1 OFFSET ...
    4. Output the queries you ran, their results, and the business rules found (or state that there are none).
    Report only numbers returned by the queries. If a query fails, fix it or return the error message.
    """,
    tools=[get_database_schema, get_business_rules_info, query_database],
    model='gpt-6.1-sol',
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
)
