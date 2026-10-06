import logging
from typing import Any
from agents import RunContextWrapper

logger = logging.getLogger(__name__)


def tool_error_message(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """
    Log the tool error and return its message to the model so it can correct its arguments.
    The SDK default hides the message behind "An error occurred while running the tool".
    """
    logger.error("Tool call failed: %s: %s", type(error).__name__, error)
    return f"Tool error ({type(error).__name__}): {error}"
