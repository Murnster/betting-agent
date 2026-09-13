#!/usr/bin/env bash
# NFL props loop — the one entry point cron calls. Everything is safe to run
# on any day: off-days exit before spending a credit, --closing spends only
# for games kicking off inside its window, grading is idempotent.
#
#   scripts/nfl_loop.sh card     game day: today's slate → picks, Discord card
#   scripts/nfl_loop.sh closing  pre-kickoff: closing prices on held picks → CLV
#   scripts/nfl_loop.sh grade    next morning: finalize games, grade, results post
#   scripts/nfl_loop.sh backup   pg_dump to backups/postgres/, prune >30 days
#
# Cron gets a minimal PATH, so uv and the claude CLI (validator) are located
# here; override with UV_BIN / CLAUDE_BIN_DIR if they live elsewhere.
# Full setup: docs/NFL_SETUP.md
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
CLAUDE_BIN_DIR="${CLAUDE_BIN_DIR:-$HOME/.local/bin}"
export PATH="$CLAUDE_BIN_DIR:$(dirname "$UV_BIN"):/usr/local/bin:/usr/bin:/bin"
# The validator shells out to `claude -p`; it must not think it is nested in
# an interactive Claude Code session.
unset CLAUDECODE

# Games to price on a full slate. Each game costs 6 Odds API credits (2
# receiving markets + TD board + 3 ladder boards). 16 is the largest slate of
# the 2026 season (Jan 10), so every game of every day gets priced: --suggest
# ranks the whole day in ONE flat list with no idea slates exist, so a smaller
# N could spend all its games on the 1pm window and leave the 4pm window and
# Sunday night with nothing to card. Pricing everything peaks at 616 credits
# in November against 1500 pooled across the three keys.
SUGGEST="${NFL_SUGGEST:-16}"
# Main-card prop candidates kept PER SLATE (props.py cuts per kickoff window,
# not per day, so the 4pm window and Sunday night keep their own allowance).
# Raising this does not add card slots — those are capped per slate — it only
# adds off-card saved picks, and off-card picks are staked in the paper ledger
# too, so it is a real change in exposure. 10 per window is the default.
MAX_PICKS="${NFL_MAX_PICKS:-10}"
# Optional: price only games kicking off within N hours (props.py
# --within-hours). Used by the early Sunday run, which exists solely for the
# 9:30 ET international game — without the window it would price the whole
# Sunday slate hours before the midday run does it again at fresher numbers.
WITHIN_HOURS="${NFL_WITHIN_HOURS:-}"
# Optional: leave the day's night window to the evening run (props.py
# --skip-night). Set on the Sunday midday run so Sunday Night is carded at
# 19:00 alongside Monday/Thursday primetime rather than eight hours early.
SKIP_NIGHT="${NFL_SKIP_NIGHT:-}"

cd "$PROJECT_ROOT"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

case "${1:-}" in
  card)
    echo "[$(stamp)] card: today's slate (up to $SUGGEST games by model heat)${WITHIN_HOURS:+, kicking off within ${WITHIN_HOURS}h}${SKIP_NIGHT:+, excluding the night window}"
    "$UV_BIN" run python scripts/props.py --today --save --suggest "$SUGGEST" \
        --max-picks "$MAX_PICKS" ${WITHIN_HOURS:+--within-hours "$WITHIN_HOURS"} \
        ${SKIP_NIGHT:+--skip-night}
    ;;
  closing)
    echo "[$(stamp)] closing: held picks kicking off inside the window"
    "$UV_BIN" run python scripts/props.py --closing --window-minutes "${NFL_CLOSING_WINDOW:-90}"
    ;;
  grade)
    echo "[$(stamp)] grade: finalize NFL games, grade props, post results"
    "$UV_BIN" run python scripts/grade.py --sport NFL
    ;;
  backup)
    echo "[$(stamp)] backup: pg_dump"
    "$UV_BIN" run python scripts/backup_db.py
    ;;
  crontab)
    # Print the schedule for `crontab -e`. Times are the MACHINE'S local time:
    # props.py --today matches kickoffs to the local calendar day, so the box
    # should run in a North American zone (this schedule assumes ET+1,
    # America/Halifax; shift the hours if you run elsewhere).
    cat <<EOF
# ---- betting-agent NFL props loop (docs/NFL_SETUP.md) ----
SHELL=/bin/bash
# A card run covers every slate of the day that has not kicked off yet
# (props.py --today cards each slate in one pass), so what matters is that a
# run fires before each window — and not so far before that it prices a game
# on stale numbers when a later run could do it for the same credits.
# Fri/Sat: one card at 12:45 local. Their games are a single unsplit slate
# whatever the hour (Christmas Friday, the December Saturday doubleheaders).
45 12 * * 5,6     $PROJECT_ROOT/scripts/nfl_loop.sh card    >> $LOG_DIR/nfl_card.log 2>&1
# Sunday afternoon: the card at 12:45 local, before the 1pm ET window, for the
# early and late windows only. Sunday Night is excluded and left to the 19:00
# run below — carding a 20:20 ET game at 11:45 ET meant an eight-hour-old
# price on the one game that could have had a two-hour-old one (user, Sep 13).
45 12 * * 0       NFL_SKIP_NIGHT=1 $PROJECT_ROOT/scripts/nfl_loop.sh card >> $LOG_DIR/nfl_card.log 2>&1
# Sunday international game (London/Dublin/Berlin/Madrid) kicks at 9:30 ET =
# 10:30 local, BEFORE the 12:45 run — which then drops it as already started,
# so it got no card at all. Its own run at 08:45 local (07:45 ET), windowed to
# the next 4 hours so it prices that one game and leaves the rest of Sunday to
# the midday run. Free on the ~12 Sundays with no international game.
45 8  * * 0       NFL_WITHIN_HOURS=4 $PROJECT_ROOT/scripts/nfl_loop.sh card >> $LOG_DIR/nfl_card.log 2>&1
# Thanksgiving is the same problem on a Thursday: 13:00 and 16:30 ET games
# that the 19:00-local run would find already kicked off. Same early run,
# windowed to 10 hours — on any other Thursday the only game is at 20:15 ET,
# 12.5 hours out, so this exits free and the 19:00 run does the work.
45 8  * * 4       NFL_WITHIN_HOURS=10 $PROJECT_ROOT/scripts/nfl_loop.sh card >> $LOG_DIR/nfl_card.log 2>&1
# Primetime, every night that has one: the card at 19:00 local (18:00 ET).
# Sunday is in the list so Sunday Night is carded here, ~2h before its 20:20 ET
# kickoff; the afternoon games have already started by then, so --today drops
# them and this run prices only the night game. Wednesday is not a typo — the
# 2026 season opens on one (Sep 9 NE@SEA) and week 12 has another. Free on any
# night with no game.
0  19 * * 0,1,2,3,4 $PROJECT_ROOT/scripts/nfl_loop.sh card  >> $LOG_DIR/nfl_card.log 2>&1
# Closing prices, hourly, every day (the events call is free and nothing is
# fetched unless a held pick kicks off inside the window)
0  9-23 * * *     $PROJECT_ROOT/scripts/nfl_loop.sh closing >> $LOG_DIR/nfl_closing.log 2>&1
# Grade every morning (stats for a Sunday land on nflverse Monday/Tuesday; re-runs are free)
0  9  * * *     $PROJECT_ROOT/scripts/nfl_loop.sh grade   >> $LOG_DIR/nfl_grade.log 2>&1
# Nightly database dump
15 3  * * *     $PROJECT_ROOT/scripts/nfl_loop.sh backup  >> $LOG_DIR/nfl_backup.log 2>&1
EOF
    ;;
  *)
    echo "usage: $0 {card|closing|grade|backup|crontab}" >&2
    exit 2
    ;;
esac
