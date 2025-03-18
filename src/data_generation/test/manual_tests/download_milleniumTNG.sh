#!/bin/bash

# Remember to update your HOME, ie: export HOME=/mnt/diskD/space/flamingo

# Load .env file
export $(grep -v '^#' .env | xargs)

# Ensure SSH credentials are set
if [ -z "$SSH_USER" ] || [ -z "$SSH_PASSWORD" ] || [ -z "$SSH_HOST" ] || [ -z "$SSH_VIRGO" ]; then
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

    echo "Downloading $REMOTE_FILE to $LOCAL_DEST..."

    # Use sshpass to supply password, and allow manual input of 2FA code when prompted
    ERROR_MESSAGE=$(sshpass -p "$SSH_PASSWORD" ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "
        ssh -i $SSH_KEY $SSH_USER@$SSH_VIRGO 'scp -r $REMOTE_FILE $LOCAL_DEST'
    " 2>&1)

    # Handle 2FA manually by checking if the output contains a 2FA prompt
    if [[ "$ERROR_MESSAGE" == *"Verification code"* ]]; then
        echo "Please enter your 2FA code:"
        read -p "2FA Code: " OTP_CODE

        # Send the OTP code to the prompt
        ERROR_MESSAGE=$(sshpass -p "$SSH_PASSWORD" ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "
            ssh -i $SSH_KEY $SSH_USER@$SSH_VIRGO 'scp -r $REMOTE_FILE $LOCAL_DEST'
        " 2>&1)
    fi

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded: $REMOTE_FILE -> $LOCAL_DEST"
    else
        # Capture and display the error message from the last command
        echo "❌ Failed to download: $REMOTE_FILE"
        echo "Error details: $?"  # This shows the exit code of the last command
        echo "Full error output:"
        echo "$ERROR_MESSAGE"
    fi
done
