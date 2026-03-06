#!/bin/bash
set -e

# Usage: ./daily_workflow.sh [morning|pregame|postgame]

# Ensure PATH includes common user bin directories for uv visibility in cron
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MODE=$1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure uv is in path or use absolute path if needed.
# Assuming uv is in the user's PATH since they are setting this up.
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH."
    exit 1
fi

# NOTE: NFL commented out during offseason to conserve Odds API quota
SPORTS=("NBA" "NHL")  # Add "NFL" back when season starts (~September)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_extraction() {
    local phase=$1
    local extra_args=""
    if [ "$phase" == "closing_smart" ]; then
        phase="closing"
        extra_args="--smart"
    fi

    for sport in "${SPORTS[@]}"; do
        log "Running extraction ($phase $extra_args) for $sport..."
        uv run python scripts/extract.py "$phase" --sport "$sport" $extra_args
    done
}

run_picks() {
    for sport in "${SPORTS[@]}"; do
        log "Generating picks for $sport..."
        uv run python scripts/picks.py --sport "$sport" --bankroll 100 --save
    done
}

case "$MODE" in
    morning)
        log "Starting Morning Routine"
        
        # 1. Grade yesterday's picks first to clear the slate
        log "Grading previous picks..."
        uv run python scripts/grade.py

        # 2. Extract morning data (schedule, opening odds, weather)
        run_extraction "morning"

        # 3. Generate today's picks
        run_picks
        ;;
        
    pregame)
        log "Starting Pre-game Routine (Closing Odds - All Games)"
        run_extraction "closing"
        ;;

    pregame_smart)
        log "Starting Pre-game Routine (Closing Odds - Smart Filter)"
        run_extraction "closing_smart"
        ;;
        
    postgame)
        log "Starting Post-game Routine (Final Scores)"
        run_extraction "postgame"
        ;;
        
    *)
        echo "Usage: $0 [morning|pregame|pregame_smart|postgame]"
        exit 1
        ;;
esac

log "Routine ($MODE) completed successfully."
