#!/bin/bash

# Directory containing FLAC files
input_dir="/Volumes/Seagate HDD/Brave/slakh processed/train/flac_stems/bass"
output_dir="/Volumes/Seagate HDD/Brave/slakh processed/train/wav_stems/bass"

# Iterate over all .flac files in the directory
for file in "$input_dir"/*.flac; do
    # Get the filename without the extension
    base_name=$(basename "$file" .flac)
    
    # Convert to WAV using ffmpeg
    ffmpeg -i "$file" -ar 44100 -ac 2 "${output_dir}/${base_name}.wav"
done

echo "Conversion complete!"

