"""
SQLAlchemy ORM models for betting agent DB schema.
Tables: games, odds, sentiment, picks
"""

from datetime import datetime
from sqlalchemy import Boolean, Column, Date, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy import TIMESTAMP as TIMESTAMPTZ
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)          # 'NFL', 'NBA', 'NHL'
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=True)               # NULL for non-NFL
    game_date = Column(Date, nullable=False)
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    home_score = Column(Integer, nullable=True)         # NULL until final
    away_score = Column(Integer, nullable=True)         # NULL until final
    neutral_site = Column(Boolean, default=False)
    weather_desc = Column(String(100), nullable=True)
    temperature_f = Column(Float, nullable=True)
    wind_mph = Column(Float, nullable=True)
    is_playoff = Column(Boolean, default=False)
    status = Column(String(20), default="scheduled")    # scheduled|final|cancelled
    external_id = Column(String(100), unique=True, nullable=True)  # from Odds API

    # NHL Box Score Stats
    home_sog = Column(Integer, nullable=True)
    away_sog = Column(Integer, nullable=True)
    home_ppg = Column(Integer, nullable=True)
    away_ppg = Column(Integer, nullable=True)
    home_ppo = Column(Integer, nullable=True)
    away_ppo = Column(Integer, nullable=True)
    home_pim = Column(Integer, nullable=True)
    away_pim = Column(Integer, nullable=True)
    home_hits = Column(Integer, nullable=True)
    away_hits = Column(Integer, nullable=True)
    home_blocks = Column(Integer, nullable=True)
    away_blocks = Column(Integer, nullable=True)
    home_faceoff_pct = Column(Float, nullable=True)
    away_faceoff_pct = Column(Float, nullable=True)

    # MLB Box Score Stats — Batting (9 stats × 2 sides = 18)
    home_hits = Column(Integer, nullable=True)
    away_hits = Column(Integer, nullable=True)
    home_runs_scored = Column(Integer, nullable=True)
    away_runs_scored = Column(Integer, nullable=True)
    home_home_runs = Column(Integer, nullable=True)
    away_home_runs = Column(Integer, nullable=True)
    home_rbi = Column(Integer, nullable=True)
    away_rbi = Column(Integer, nullable=True)
    home_strikeouts = Column(Integer, nullable=True)
    away_strikeouts = Column(Integer, nullable=True)
    home_walks = Column(Integer, nullable=True)
    away_walks = Column(Integer, nullable=True)
    home_doubles = Column(Integer, nullable=True)
    away_doubles = Column(Integer, nullable=True)
    home_triples = Column(Integer, nullable=True)
    away_triples = Column(Integer, nullable=True)
    home_stolen_bases = Column(Integer, nullable=True)
    away_stolen_bases = Column(Integer, nullable=True)
    # MLB Box Score Stats — Pitching (5 stats × 2 sides = 10)
    home_earned_runs = Column(Integer, nullable=True)
    away_earned_runs = Column(Integer, nullable=True)
    home_pitching_strikeouts = Column(Integer, nullable=True)
    away_pitching_strikeouts = Column(Integer, nullable=True)
    home_pitching_walks = Column(Integer, nullable=True)
    away_pitching_walks = Column(Integer, nullable=True)
    home_hits_allowed = Column(Integer, nullable=True)
    away_hits_allowed = Column(Integer, nullable=True)
    home_innings_pitched = Column(Float, nullable=True)
    away_innings_pitched = Column(Float, nullable=True)

    odds = relationship("Odds", back_populates="game", cascade="all, delete-orphan")
    sentiment = relationship("Sentiment", back_populates="game", cascade="all, delete-orphan")
    picks = relationship("Pick", back_populates="game", cascade="all, delete-orphan")
    agent_validations = relationship(
        "AgentValidation", back_populates="game", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Game {self.away_team}@{self.home_team} {self.game_date} {self.sport}>"


class Odds(Base):
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    bookmaker = Column(String(50), nullable=True)
    bet_type = Column(String(20), nullable=True)        # 'moneyline'|'spread'|'total'|'prop'
    market_key = Column(String(100), nullable=True)     # Odds API market key
    description = Column(String(200), nullable=True)    # e.g. player name for props
    home_price = Column(Integer, nullable=True)         # American odds
    away_price = Column(Integer, nullable=True)
    spread_home = Column(Float, nullable=True)          # NULL for non-spread bets
    total_line = Column(Float, nullable=True)           # NULL for non-total bets
    fetched_at = Column(TIMESTAMPTZ, default=datetime.utcnow)
    is_closing = Column(Boolean, default=False)         # TRUE for closing line (CLV)

    game = relationship("Game", back_populates="odds")

    def __repr__(self) -> str:
        return f"<Odds game_id={self.game_id} {self.bet_type} {self.bookmaker}>"


class Sentiment(Base):
    __tablename__ = "sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    team = Column(String(50), nullable=True)
    source = Column(String(100), nullable=True)         # news article URL or description
    sentiment_score = Column(Float, nullable=True)      # -1.0 to 1.0
    summary = Column(Text, nullable=True)
    analysis_type = Column(String(30), default="injury") # injury | game | line_movement
    model_used = Column(String(50), nullable=True)      # e.g. 'phi4', 'llama3'
    analyzed_at = Column(TIMESTAMPTZ, default=datetime.utcnow)

    game = relationship("Game", back_populates="sentiment")

    def __repr__(self) -> str:
        return f"<Sentiment game_id={self.game_id} team={self.team} score={self.sentiment_score}>"


class Pick(Base):
    __tablename__ = "picks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    sport = Column(String(10), nullable=False)
    pick_date = Column(Date, nullable=False)
    bet_type = Column(String(20), nullable=False)       # 'moneyline'|'spread'|'total'|'prop'
    pick_side = Column(String(100), nullable=False)     # team name or 'over'/'under'
    model_prob = Column(Float, nullable=False)          # our model's probability
    implied_prob = Column(Float, nullable=False)        # market implied probability
    edge = Column(Float, nullable=False)                # model_prob - implied_prob
    odds = Column(Integer, nullable=False)              # American odds at pick time
    kelly_fraction = Column(Float, nullable=True)
    recommended_bet = Column(Float, nullable=True)      # dollar amount
    bankroll_at_pick = Column(Float, nullable=True)
    result = Column(String(10), nullable=True)          # 'win'|'loss'|'push'|NULL
    closing_odds = Column(Integer, nullable=True)       # for CLV calculation
    clv = Column(Float, nullable=True)                  # closing line value
    pnl = Column(Float, nullable=True)                  # profit/loss in dollars
    graded_at = Column(TIMESTAMPTZ, nullable=True)

    game = relationship("Game", back_populates="picks")

    def __repr__(self) -> str:
        return f"<Pick game_id={self.game_id} {self.bet_type} {self.pick_side} edge={self.edge:.3f}>"


class AgentValidation(Base):
    __tablename__ = "agent_validations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=True)
    external_id = Column(String(100), nullable=True)
    pick_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)
    bet_type = Column(String(20), nullable=False)
    pick_side = Column(String(100), nullable=False)
    verdict = Column(String(12), nullable=False)
    original_edge = Column(Float, nullable=True)
    adjusted_edge = Column(Float, nullable=True)
    reasons_json = Column(JSON, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    validated_at = Column(TIMESTAMPTZ, default=datetime.utcnow)

    game = relationship("Game", back_populates="agent_validations")

    def __repr__(self) -> str:
        return (
            f"<AgentValidation sport={self.sport} pick={self.pick_side} "
            f"verdict={self.verdict}>"
        )
