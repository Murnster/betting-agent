"""Discord webhook notifications for picks and results."""

from betting_agent.notifications.discord import (
    is_discord_configured,
    send_alltime_to_discord,
    send_picks_to_discord,
    send_results_to_discord,
)

__all__ = [
    "is_discord_configured",
    "send_alltime_to_discord",
    "send_picks_to_discord",
    "send_results_to_discord",
]
