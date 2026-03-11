"""
Central configuration via pydantic-settings.
All values can be overridden by environment variables or a .env file.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load .env into os.environ so dynamic lookups (e.g. Discord webhook URLs
# resolved by sport name) see these values too — not just pydantic fields.
load_dotenv(override=False)


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
    sentiment_weight: float = Field(default=0.02, description="Max sentiment edge adjustment")

    # Model thresholds
    ml_fav_threshold: float = Field(default=0.54, description="ML favorite bet threshold")
    ml_dog_threshold: float = Field(default=0.46, description="ML underdog bet threshold")
    ou_threshold: float = Field(default=5.0, description="O/U distance threshold (points)")

    # NBA API
    nba_api_rate_limit: float = Field(default=0.6, description="Seconds between nba_api calls")

    # NHL API
    nhl_api_rate_limit: float = Field(default=0.5, description="Seconds between nhlpy calls")
    nhl_api_concurrency: int = Field(default=10, description="ThreadPoolExecutor workers for NHL API calls")

    # MLB API
    mlb_api_rate_limit: float = Field(default=0.5, description="Seconds between mlbstatsapi calls")

    # Discord
    discord_enabled: bool = Field(default=True, description="Enable Discord webhook notifications")

    # Paths
    saved_models_dir: str = Field(default="saved_models", description="Directory for saved models")


# Module-level singleton — import `settings` everywhere
settings = Settings()
