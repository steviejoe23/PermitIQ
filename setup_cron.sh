#!/bin/bash
# Install PermitIQ weekly update cron job
# Runs every Sunday at 2am
#
# Usage:
#   ./setup_cron.sh          # Install the cron job
#   ./setup_cron.sh --remove # Remove the cron job

SCRIPT_DIR="/Users/stevenspero/Desktop/Boston Zoning Project"
CRON_CMD='0 2 * * 0 cd "/Users/stevenspero/Desktop/Boston Zoning Project" && ./auto_weekly_update.sh >> logs/cron_output.log 2>&1'

if [ "$1" = "--remove" ]; then
    crontab -l 2>/dev/null | grep -v "auto_weekly_update" | crontab -
    echo "Cron job removed."
    echo "Verify with: crontab -l"
    exit 0
fi

# Remove any existing entry, then add the new one
(crontab -l 2>/dev/null | grep -v "auto_weekly_update"; echo "$CRON_CMD") | crontab -

echo "Cron job installed successfully."
echo ""
echo "  Schedule: Every Sunday at 2:00 AM"
echo "  Script:   $SCRIPT_DIR/auto_weekly_update.sh"
echo "  Logs:     $SCRIPT_DIR/logs/"
echo ""
echo "Verify with:  crontab -l"
echo "Remove with:  ./setup_cron.sh --remove"
echo ""
echo "NOTE: macOS may prompt for cron permissions in System Settings > Privacy & Security > Full Disk Access."
echo "      You may need to grant /usr/sbin/cron access for the job to run."
