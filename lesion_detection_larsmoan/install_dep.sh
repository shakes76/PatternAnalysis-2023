#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

conda activate torch-gpu

# Read the requirements.txt file and try to install packages using conda
while read package; do
    if [[ ! -z "$package" && ! "$package" =~ ^\# ]]; then
        conda install --yes "$package" || echo "Failed to install $package with conda. Consider installing manually."
    fi
done < yolov7/requirements.txt

echo "Installation finished."
