"""
NFL slates: which card a game belongs on.

A slate is a kickoff window — Thursday night, the Sunday 9:30 ET
international game, Sunday early, Sunday late, Sunday night, Monday night
(plus the odd Saturday/Friday game). Cards are built per slate: a single-game slate (primetime) carries
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

# A primetime slate is one game, so its card is the only card that day and
# there is nothing to spread across. The walk-forward eval (2024-25, ranked by
# edge within the week) finds no reliable ordering inside the top ten — rank
# 1-3 realised 64.3% ±8.4 against 69.3% ±4.4 for ranks 1-10 — so the two extra
# slots cost nothing in pick quality. User's call, Sep 10 2026.
PRIMETIME_PROP_CAP = 4
WINDOW_PROP_CAP = 3
PRIMETIME_LEAN_CAP = 1
WINDOW_LEAN_CAP = 3
#: Sunday kickoffs before this ET hour are international games. No domestic
#: Sunday game starts before 13:00 ET; the international ones start at 9:30.
INTERNATIONAL_BEFORE_ET = 12
#: Kickoffs at or after this ET hour open the night window on a day that also
#: has afternoon windows (Sunday, Thanksgiving). Monday and the Wednesday
#: opener kick at 20:15-20:35 ET but are their day's only game, so they are a
#: single unsplit slate and this boundary never applies to them.
NIGHT_FROM_ET = 19


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
        if k.hour < INTERNATIONAL_BEFORE_ET:
            # London/Dublin/Berlin/Madrid kick at 9:30 ET, hours before the
            # 1pm window. It is a standalone game with nothing to share the
            # card with, exactly like Thursday or Monday night, so it gets its
            # own slate and (being single-game) the primetime caps. Defined by
            # kickoff hour rather than the schedule's `location`, which says
            # "Home" for the games where a US team is the designated host.
            part, label = "0-intl", "International"
        elif k.hour < 15:
            part, label = "1-early", "Sunday Early"
        elif k.hour < 19:
            part, label = "2-late", "Sunday Late"
        else:
            part, label = "3-night", "Sunday Night"
    elif day == "thu":
        # Thanksgiving is three standalone games (13:00 / 16:30 / 20:20 ET),
        # not one night game. Splitting them the way Sunday is split gives
        # each its own card, and stops the afternoon pair being labelled
        # "Thursday Night". Every other Thursday has only the 20:15 game, so
        # this collapses to the single primetime slate as before.
        if k.hour < 15:
            part, label = "1-early", "Thursday Early"
        elif k.hour < 19:
            part, label = "2-late", "Thursday Late"
        else:
            part, label = "3-night", "Thursday Night"
    elif day == "mon":
        part, label = "0", "Monday Night"
    else:
        part, label = "0", k.strftime("%A")
    return f"{k.date().isoformat()}-{part}", label, k.date()


def is_night_slate(commence_time: str | datetime) -> bool:
    """
    Whether a kickoff belongs to the night window of a multi-window day.

    True for Sunday Night and Thanksgiving night, false for everything else
    including Monday/Wednesday night — those are their day's only game, so
    `slate_for` never splits them out and their card run already fires in the
    evening.

    The Sunday midday card run passes `--skip-night` so Sunday Night is left
    to the 19:00 run and priced two hours out instead of eight, like every
    other primetime game (user's call, Sep 13 2026).
    """
    return slate_for(commence_time)[0].endswith("-3-night")


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


def cap_per_slate(candidates: list[BetCandidate], slates: list[Slate],
                  cap: int) -> list[BetCandidate]:
    """
    The best `cap` candidates *in each slate*, in the input order.

    Cards are built per slate, so a single global cut by edge is not safe: the
    day's highest edges cluster, and a Sunday whose top candidates all sit in
    the 1pm window leaves the 4pm window and Sunday night with nothing to card
    — and a slate with nothing on it is not posted at all. Cutting per slate
    gives every kickoff window its own allowance.

    A candidate belonging to no slate (an event with no commence_time) is kept:
    it can never be carded, but it is still a saved paper pick and dropping it
    here would be a silent behaviour change.
    """
    keep: set[int] = set()
    matched: set[int] = set()
    for slate in slates:
        mine = [i for i, c in enumerate(candidates) if _in_slate(c, slate)]
        matched.update(mine)
        mine.sort(key=lambda i: candidates[i].edge, reverse=True)
        keep.update(mine[:cap])
    keep |= set(range(len(candidates))) - matched
    return [c for i, c in enumerate(candidates) if i in keep]


def candidates_in_slate(candidates: list[BetCandidate], slate: Slate) -> list[BetCandidate]:
    return [c for c in candidates if _in_slate(c, slate)]
