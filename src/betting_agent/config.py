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
    # Each key carries its own monthly quota (500 on the free tier). The NFL
    # props loop's November peak is ~564 credits, so one key cannot cover the
    # season; the client rotates to the next key when one is exhausted.
    odds_api_key_2: str = Field(default="", description="Fallback Odds API key")
    odds_api_key_3: str = Field(default="", description="Second fallback Odds API key")
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
    prop_fallback_bookmakers: str = Field(
        default="draftkings,fanduel",
        description=(
            "Ordered Odds API bookmaker keys to price NFL props against when the "
            "preferred book posts none. bet365 is not in the API's player-prop "
            "feed, so without a fallback the props loop never produces a pick."
        ),
    )
    # Anytime-touchdown scorers (sports/nfl/td_props.py). Fetched for one
    # extra credit per game; the best scorer per game sits on the card like
    # the game lean (PICK inside the edge window, else LEAN at stake 0).
    td_props_enabled: bool = Field(default=True)
    td_props_per_game: int = Field(default=2)
    # Ladder hits — the "best overs" section: the player reaching a milestone
    # (60+ receiving yds, 6+ receptions, 40+ rushing) priced on the books'
    # alternate boards. One extra credit per game per market. Tracked as its
    # own paper book (Pick.strategy = "ladder") against its own bankroll.
    ladder_enabled: bool = Field(default=True)
    ladder_bankroll: float = Field(default=100.0, description="Ladder section's paper bankroll")
    # Straight overs — the book's main-line Over on the receiving markets,
    # at least one per game (user, Sep 8 2026), own paper book
    # (Pick.strategy = "overs"). No extra credits: the main markets are
    # already fetched for the unders card.
    overs_enabled: bool = Field(default=True)
    overs_bankroll: float = Field(default=100.0, description="Straight-overs section's paper bankroll")
    ladder_markets: str = Field(
        default="player_receptions,player_reception_yds,player_rush_yds",
        description=(
            "Base markets whose alternate boards the ladder prices (1 credit per game "
            "each). All three are on: the pool is an experiment by the user's choice, "
            "even though the receptions ladder over-claims by 10-20pp in the diagnostic."
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
        description=(
            "Validator model, prefixed by provider: 'gemini/<model>' calls the "
            "Gemini API, 'claude/<model>' runs the local `claude -p` CLI"
        ),
    )
    agent_shadow: bool = Field(
        default=True,
        description=(
            "Shadow mode: record validator verdicts (agent_validations, pick "
            "cards, Discord) without changing edge, sizing, or dropping picks"
        ),
    )
    agent_claude_web_search: bool = Field(
        default=False, description="Allow the claude CLI validator to use WebSearch"
    )
    agent_claude_max_turns: int = Field(
        default=4, description="Agentic turn cap per claude CLI validator call"
    )
    agent_claude_max_call_usd: float = Field(
        default=0.25, description="Hard per-call spend cap passed to claude --max-budget-usd"
    )
    agent_retries: int = Field(
        default=1,
        description=(
            "Extra attempts when a validator call fails or is killed on its cap — a "
            "skipped validation should never come into play (user, Sep 2026)"
        ),
    )
    agent_claude_timeout: int = Field(
        default=180,
        description=(
            "Seconds to wait for one `claude -p` validator call. A multi-pick game "
            "with WebSearch runs 40-60s; a kill here loses the spend unrecorded."
        ),
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
    def odds_api_keys(self) -> list[str]:
        """
        Every configured Odds API key, primary first, blanks and duplicates
        dropped. `OddsAPIClient` walks this list when a key's monthly quota
        runs out, so the loop keeps running instead of dying with
        "No prop odds returned" partway through November.
        """
        keys: list[str] = []
        for key in (self.odds_api_key, self.odds_api_key_2, self.odds_api_key_3):
            key = (key or "").strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    @property
    def preferred_bookmaker_list(self) -> list[str]:
        """preferred_bookmakers parsed into keys ([] when unset)."""
        return [b.strip() for b in self.preferred_bookmakers.split(",") if b.strip()]

    @property
    def prop_fallback_bookmaker_list(self) -> list[str]:
        """prop_fallback_bookmakers parsed into keys ([] when unset)."""
        return [b.strip() for b in self.prop_fallback_bookmakers.split(",") if b.strip()]


# Module-level singleton — import `settings` everywhere
settings = Settings()
