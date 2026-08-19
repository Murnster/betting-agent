"""
Central configuration via pydantic-settings.
All values can be overridden by environment variables or a .env file.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load .env into os.environ so dynamic lookups (e.g. Discord webhook URLs
# resolved by sport name) see these values too — not just pydantic fields.
# Preserve standard precedence: explicitly exported environment variables win.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/betting_agent",
        description="PostgreSQL connection URL",
    )

    # The Odds API
    odds_api_key: str = Field(default="", description="The Odds API key")
    odds_api_base: str = Field(
        default="https://api.the-odds-api.com/v4/sports",
        description="The Odds API base URL",
    )

    preferred_bookmakers: str = Field(
        default="",
        description=(
            "Comma-separated Odds API bookmaker keys to price against "
            "(e.g. 'bet365'). Empty means shop every book in the response."
        ),
    )

    # OpenWeatherMap
    weather_api_key: str = Field(default="", description="OpenWeatherMap API key")

    # Ollama (Phase 2)
    ollama_url: str = Field(
        default="http://localhost:11434/api/generate",
        description="Ollama API endpoint",
    )
    ollama_model: str = Field(default="llama3.1:8b", description="Ollama model to use")
    ollama_timeout: int = Field(default=60, description="Ollama request timeout in seconds")
    sentiment_depth: str = Field(default="full", description="Context depth: basic (injuries only) or full")

    # Strategy constants
    min_edge_pct: float = Field(default=0.015, description="Minimum edge % to place a bet")
    max_kelly_pct: float = Field(default=0.05, description="Max Kelly fraction cap")
    max_bet_pct: float = Field(default=0.07, description="Max % of bankroll per bet")
    min_bet_pct: float = Field(default=0.0, description="Min % of bankroll per bet")
    starting_bankroll: float = Field(default=100.0, description="Default starting bankroll")
    max_picks_per_run: int = Field(default=3, description="Maximum picks to return per sport run")
    sentiment_weight: float = Field(default=0.02, description="Max sentiment edge adjustment")

    # Model thresholds
    ml_fav_threshold: float = Field(default=0.54, description="ML favorite bet threshold")
    ml_dog_threshold: float = Field(default=0.46, description="ML underdog bet threshold")
    ou_threshold: float = Field(default=5.0, description="O/U distance threshold (points)")

    # Guardrails — reject structurally suspect picks
    max_edge_pct: float = Field(default=0.15, description="Reject picks with edge > 15%")
    max_underdog_odds: int = Field(default=500, description="Reject ML picks worse than +500")
    min_model_prob: float = Field(default=0.20, description="Reject ML picks with model prob < 20%")

    # NBA API
    nba_api_rate_limit: float = Field(default=0.6, description="Seconds between nba_api calls")

    # NHL API
    nhl_api_rate_limit: float = Field(default=0.5, description="Seconds between nhlpy calls")
    nhl_api_concurrency: int = Field(default=10, description="ThreadPoolExecutor workers for NHL API calls")

    # MLB API
    mlb_api_rate_limit: float = Field(default=0.5, description="Seconds between mlbstatsapi calls")

    # Discord
    discord_enabled: bool = Field(default=True, description="Enable Discord webhook notifications")

    # Lean validator
    gemini_api_key: str = Field(default="", description="Gemini API key for validator requests")
    tavily_api_key: str = Field(default="", description="Tavily API key for validator search")
    agent_enabled: bool = Field(default=False, description="Enable the post-picks validator")
    agent_mode: str = Field(default="top", description="Validator mode: off, top, or all")
    agent_max_games_per_run: int = Field(
        default=5, description="Maximum number of games to validate per run"
    )
    agent_search_queries_per_game: int = Field(
        default=2, description="Maximum Tavily queries per validated game"
    )
    agent_model: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Default Gemini model for validator reasoning",
    )
    agent_premium_model: str = Field(
        default="gemini/gemini-2.5-pro",
        description="Premium Gemini model for future escalation",
    )
    agent_enable_premium_escalation: bool = Field(
        default=False, description="Allow premium-model escalation for validator reasoning"
    )
    agent_premium_max_games_per_day: int = Field(
        default=1, description="Maximum premium-model validations per day"
    )
    agent_max_edge_adjustment: float = Field(
        default=0.03, description="Maximum absolute edge adjustment from the validator"
    )
    agent_daily_budget_usd: float = Field(
        default=0.50, description="Daily validator budget before skipping validation"
    )
    agent_monthly_budget_target_usd: float = Field(
        default=15.0, description="Target monthly validator budget for observability"
    )
    agent_request_timeout: int = Field(
        default=20, description="Validator API request timeout in seconds"
    )
    agent_request_retries: int = Field(
        default=2, description="Number of retries for transient validator API failures"
    )
    agent_request_retry_backoff_seconds: float = Field(
        default=1.0, description="Base backoff in seconds between validator retries"
    )

    # Paths
    saved_models_dir: str = Field(default="saved_models", description="Directory for saved models")

    @property
    def preferred_bookmaker_list(self) -> list[str]:
        """preferred_bookmakers parsed into keys ([] when unset)."""
        return [b.strip() for b in self.preferred_bookmakers.split(",") if b.strip()]


# Module-level singleton — import `settings` everywhere
settings = Settings()
