#!/bin/bash

# Usage: ./quick_results.sh [yesterday|season] [sport]

MODE=${1:-yesterday}
SPORT=${2:-NBA}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Determine current season year (heuristic: if month > 6, use current year, else prev year)
CURRENT_YEAR=$(date +%Y)
MONTH=$(date +%m)
if [ "$MONTH" -lt 7 ]; then
    CURRENT_SEASON=$((CURRENT_YEAR - 1))
else
    CURRENT_SEASON=$CURRENT_YEAR
fi

echo "========================================"
echo "  BETTING AGENT QUICK RESULTS"
echo "========================================"

if [ "$MODE" == "yesterday" ]; then
    YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
    echo "  Mode: Yesterday's Results ($YESTERDAY)"
    echo "  Sport: $SPORT"
    echo "----------------------------------------"
    # Filter picks by date. Since report.py aggregates, we might need a direct query or
    # we can use grade.py output if it shows recent grades, but report.py is better for summary.
    # report.py currently aggregates 'since' a date.
    
    # We'll use a python one-liner or update report.py to support --date.
    # For now, let's use report.py with --since yesterday
    uv run python -c "from betting_agent.accounting.roi import format_roi_report; from datetime import date, timedelta; yesterday = date.today() - timedelta(days=1); print(format_roi_report(sport='$SPORT', since=yesterday))"

elif [ "$MODE" == "season" ]; then
    echo "  Mode: Current Season ($CURRENT_SEASON)"
    echo "  Sport: $SPORT"
    echo "----------------------------------------"
    uv run python scripts/report.py --sport "$SPORT" --season "$CURRENT_SEASON"

else
    echo "Usage: ./quick_results.sh [yesterday|season] [sport]"
    echo "Example: ./quick_results.sh yesterday NBA"
    echo "Example: ./quick_results.sh season NFL"
fi
