"""
NFL slates: which card a game belongs on.

A slate is a kickoff window — Thursday night, Sunday early, Sunday late,
Sunday night, Monday night (plus the odd Saturday/Friday/international
game). Cards are built per slate: a single-game slate (primetime) carries
one game lean and up to PRIMETIME_PROP_CAP props; a multi-game window carries
up to WINDOW_LEAN_CAP leans and WINDOW_PROP_CAP props. Everything that
clears the floors is still saved for the paper trade; the caps only decide
what goes on the card (Pick.on_card).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from betting_agent.intelligence.picks import BetCandidate

ET = ZoneInfo("America/New_York")

PRIMETIME_PROP_CAP = 2
WINDOW_PROP_CAP = 3
PRIMETIME_LEAN_CAP = 1
WINDOW_LEAN_CAP = 3


def _kickoff_et(commence_time: str | datetime) -> datetime:
    if isinstance(commence_time, str):
        commence_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    if commence_time.tzinfo is None:
        commence_time = commence_time.replace(tzinfo=timezone.utc)
    return commence_time.astimezone(ET)


def slate_for(commence_time: str | datetime) -> tuple[str, str, date]:
    """(sort key, label, ET date) for a kickoff. Windows follow Eastern time."""
    k = _kickoff_et(commence_time)
    day = k.strftime("%a").lower()
    if day == "sun":
        if k.hour < 15:
            part, label = "1-early", "Sunday Early"
        elif k.hour < 19:
            part, label = "2-late", "Sunday Late"
        else:
            part, label = "3-night", "Sunday Night"
    elif day == "thu":
        part, label = "0", "Thursday Night"
    elif day == "mon":
        part, label = "0", "Monday Night"
    else:
        part, label = "0", k.strftime("%A")
    return f"{k.date().isoformat()}-{part}", label, k.date()


@dataclass
class Slate:
    key: str
    label: str
    date: date
    events: list[dict] = field(default_factory=list)

    @property
    def event_ids(self) -> set[str]:
        return {str(e.get("id")) for e in self.events if e.get("id")}

    @property
    def single_game(self) -> bool:
        return len(self.events) == 1

    @property
    def prop_cap(self) -> int:
        return PRIMETIME_PROP_CAP if self.single_game else WINDOW_PROP_CAP

    @property
    def lean_cap(self) -> int:
        return PRIMETIME_LEAN_CAP if self.single_game else WINDOW_LEAN_CAP

    def title(self) -> str:
        if self.single_game:
            e = self.events[0]
            return f"{self.label} — {e.get('away_team', '?')} @ {e.get('home_team', '?')}"
        return f"{self.label} — {len(self.events)} games"


def group_events_by_slate(events: list[dict]) -> list[Slate]:
    """Events (Odds API shape: id, commence_time, home/away) → slates in kickoff order."""
    slates: dict[str, Slate] = {}
    for e in events:
        ct = e.get("commence_time")
        if not ct:
            continue
        key, label, day = slate_for(ct)
        slates.setdefault(key, Slate(key, label, day)).events.append(e)
    for s in slates.values():
        s.events.sort(key=lambda e: e.get("commence_time") or "")
    return [slates[k] for k in sorted(slates)]


def _in_slate(c: BetCandidate, slate: Slate) -> bool:
    return str(c.external_id) in slate.event_ids


def select_card(candidates: list[BetCandidate], slate: Slate, cap: int) -> list[BetCandidate]:
    """
    The top `cap` candidates (by edge) in this slate. Marks them
    extra["card"]=True; the rest stay saved but off the card.
    """
    mine = sorted((c for c in candidates if _in_slate(c, slate)),
                  key=lambda c: c.edge, reverse=True)
    card = mine[:cap]
    for c in card:
        c.extra["card"] = True
    return card


def candidates_in_slate(candidates: list[BetCandidate], slate: Slate) -> list[BetCandidate]:
    return [c for c in candidates if _in_slate(c, slate)]
