#!/bin/bash

# Remember to update your HOME, ie: export HOME=/mnt/diskD/space/flamingo

# Load .env file
export $(grep -v '^#' .env | xargs)

# Ensure SSH credentials are set
if [ -z "$SSH_USER" ] || [ -z "$SSH_PASSWORD" ] || [ -z "$SSH_KEY" ] || [ -z "$SSH_HOST" ]; then
    echo "Error: SSH credentials are not set in .env"
    exit 1
fi

# List of files to copy
FILES=(
    # L1000N1800
    "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0073.hdf5"
    "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/flamingo_0073.hdf5"
    # L2800N5040
    "/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0073.hdf5"
    "/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP-HBT/flamingo_0073.hdf5"
    # L1000N0900
    "/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0073.hdf5"
    "/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/SOAP-HBT/flamingo_0073.hdf5"
    # L1000N3600
    "/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0073.hdf5"
    "/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/SOAP-HBT/flamingo_0073.hdf5"
)

# Corresponding local destinations
DESTINATIONS=(
    # L1000N1800
    "$HOME/L1000N1800/halo_properties_0073.hdf5"
    "$HOME/L1000N1800/flamingo_0073.hdf5"
    # L2800N5040
    "$HOME/L2800N5040/halo_properties_0073.hdf5"
    "$HOME/L2800N5040/flamingo_0073.hdf5"
    # L1000N0900
    "$HOME/L1000N0900/halo_properties_0073.hdf5"
    "$HOME/L1000N0900/flamingo_0073.hdf5"
    # L1000N3600
    "$HOME/L1000N3600/halo_properties_0073.hdf5"
    "$HOME/L1000N3600/flamingo_0073.hdf5"
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

    # Capture the output of the scp command and any errors
    ERROR_MESSAGE=$(sshpass -p "$SSH_PASSWORD" scp -i "$SSH_KEY" "$SSH_USER@$SSH_HOST:$REMOTE_FILE" "$LOCAL_DEST" 2>&1)

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
