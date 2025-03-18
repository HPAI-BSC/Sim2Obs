#!/bin/bash

# Remember to update your HOME, ie: export HOME=/mnt/diskD/space/flamingo

# Load .env file
export $(grep -v '^#' .env | xargs)

# Ensure SSH credentials are set
if [ -z "$SSH_USER" ] || [ -z "$SSH_PASSWORD" ] || [ -z "$SSH_HOST" ] || [ -z "$SSH_VIRGO" ] || [ -z "$SSH_KEY" ]; then
    echo "Error: SSH credentials are not set in .env"
    exit 1
fi

# List of files to copy
FILES=(
    # MTNG-L125-1080-A
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L125-1080-A/output/groups_196"
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L125-1080-A/output/snapdir_196"
    # MTNG-L31.25-270-A
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L31.25-270-A/output/groups_196"
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L31.25-270-A/output/snapdir_196"
    # MTNG-L500-4320-A
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/groups_196"
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/snapdir_196"
    # MTNG-L62.5-540-A
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L62.5-540-A/output/groups_196"
    "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L62.5-540-A/output/snapdir_196"
)

# Corresponding local destinations
DESTINATIONS=(
    # MTNG-L125-1080-A
    "$HOME/MTNG-L125-1080-A/output/groups_196"
    "$HOME/MTNG-L125-1080-A/output/snapdir_196"
    # MTNG-L31.25-270-A
    "$HOME/MTNG-L31.25-270-A/output/groups_196"
    "$HOME/MTNG-L31.25-270-A/output/snapdir_196"
    # MTNG-L500-4320-A
    "$HOME/MTNG-L500-4320-A/output/groups_196"
    "$HOME/MTNG-L500-4320-A/output/snapdir_196"
    # MTNG-L62.5-540-A
    "$HOME/MTNG-L62.5-540-A/output/groups_196"
    "$HOME/MTNG-L62.5-540-A/output/snapdir_196"
)

# Check if both arrays have the same length
if [ ${#FILES[@]} -ne ${#DESTINATIONS[@]} ]; then
    echo "Error: FILES and DESTINATIONS lists must have the same length."
    exit 1
fi

# Loop through each file and copy it to the corresponding destination
for i in "${!FILES[@]}"; do
    REMOTE_FILE="${FILES[$i]}"
    LOCAL_DEST="${DESTINATIONS[$i]}"

    LOG_FILE="$HOME/download_${i}.log"  # Log file for each file download
    echo "Downloading $REMOTE_FILE to $LOCAL_DEST..."

    # Run the SCP command with nohup, redirecting both stdout and stderr to the log file
    nohup sshpass -p "$SSH_PASSWORD" ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "
        ssh -i $SSH_KEY $SSH_USER@$SSH_VIRGO 'scp -r $REMOTE_FILE $LOCAL_DEST'
    " > "$LOG_FILE" 2>&1 &  # Run the command in the background, log output to $LOG_FILE

    # Notify user that the download is running in the background
    echo "Download is running in the background. Check $LOG_FILE for progress."

    # Optionally,  monitor the log or notify if there was a 2FA prompt
    # However, handling manual 2FA input automatically with sshpass is not feasible
done