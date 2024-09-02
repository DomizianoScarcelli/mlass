#!/bin/bash

# Number of times to attempt restarting the script
max_restarts=40

# Path to your Python script
script="lass_audio.lass.graphical_model_separate_three_sources"

# Initialize the counter
counter=0

# Loop to restart the script if it crashes
while [ $counter -lt $max_restarts ]
do
    echo "Starting script, attempt $((counter+1)) of $max_restarts..."
    
    # Run the Python script
    python -m $script
    
    # Check if the script exited with a non-zero status (indicating a crash)
    if [ $? -ne 0 ]; then
        echo "Script crashed, restarting..."
        counter=$((counter+1))
    else
        echo "Script finished successfully."
        break
    fi
done

if [ $counter -eq $max_restarts ]; then
    echo "Script crashed $max_restarts times. Stopping."
fi
