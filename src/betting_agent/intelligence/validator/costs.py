from __future__ import annotations

import logging
from datetime import date

from betting_agent.config import settings
from betting_agent.db.queries import get_agent_spend_for_date
from betting_agent.db.session import get_session

logger = logging.getLogger(__name__)

_MODEL_RATES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    model_name = model.split("/", 1)[-1]
    input_rate, output_rate = _MODEL_RATES.get(model_name, _MODEL_RATES["gemini-2.5-flash"])
    return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)


def get_daily_spend(target_date: date) -> float:
    try:
        with get_session() as session:
            return get_agent_spend_for_date(session, target_date)
    except Exception as exc:
        logger.debug("Could not load validator spend for %s: %s", target_date, exc)
        return 0.0


def budget_allows(target_date: date, run_cost: float) -> bool:
    return get_daily_spend(target_date) + run_cost < settings.agent_daily_budget_usd
