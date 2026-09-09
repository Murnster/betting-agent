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

cd "$PROJECT_ROOT"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

case "${1:-}" in
  card)
    echo "[$(stamp)] card: today's slate (up to $SUGGEST games by model heat)"
    "$UV_BIN" run python scripts/props.py --today --save --suggest "$SUGGEST" \
        --max-picks "$MAX_PICKS"
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
# One card run per day covers every slate that day (props.py --today cards each
# slate in one pass and skips games already kicked off), so the only thing that
# matters is running before the day's FIRST kickoff.
# Sun/Fri/Sat: the card at 12:45 local, before the 1pm ET window (Christmas
# Friday and the December Saturday doubleheaders start in the afternoon).
45 12 * * 0,5,6   $PROJECT_ROOT/scripts/nfl_loop.sh card    >> $LOG_DIR/nfl_card.log 2>&1
# Mon/Tue/Wed/Thu primetime: the card at 19:00 local. Wednesday is not a typo —
# the 2026 season opens on one (Sep 9 NE@SEA) and week 12 has another.
0  19 * * 1,2,3,4 $PROJECT_ROOT/scripts/nfl_loop.sh card    >> $LOG_DIR/nfl_card.log 2>&1
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
