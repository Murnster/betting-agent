#!/bin/bash
set -e

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW_SCRIPT="$PROJECT_ROOT/scripts/daily_workflow.sh"
LOG_DIR="$PROJECT_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "To automate the betting agent, add the following lines to your crontab:"
echo ""
echo "# Betting Agent Daily Workflow"
echo "SHELL=/bin/bash"
echo "# 1. Morning Routine (08:00): Grade yesterday, Extract Morning Odds, Generate Picks"
echo "0 8 * * * $WORKFLOW_SCRIPT morning >> $LOG_DIR/morning.log 2>&1"
echo ""
echo "# 2. Smart Pre-game (Every 2 hours 13:00-21:00): Capture Closing Odds for games starting soon"
echo "0 13,15,17,19,21 * * * $WORKFLOW_SCRIPT pregame_smart >> $LOG_DIR/pregame_smart.log 2>&1"
echo ""
echo "# 3. Pre-game Backup (17:30): Ensure odds captured for standard evening slate"
echo "30 17 * * * $WORKFLOW_SCRIPT pregame >> $LOG_DIR/pregame.log 2>&1"
echo ""
echo "# 4. Post-game Routine (02:00): Capture Final Scores"
echo "0 2 * * * $WORKFLOW_SCRIPT postgame >> $LOG_DIR/postgame.log 2>&1"
echo ""
echo "---------------------------------------------------------"
echo "To edit your crontab, run: crontab -e"
echo "Then paste the lines above at the bottom of the file."
echo "Logs will be written to: $LOG_DIR"
