bash#!/bin/bash
# Create conda environment
conda env create -f rave231forksrc.yml

# Activate environment
conda activate RAVESRC2

# Install RAVE in editable mode
pip install -e .

echo "Environment setup complete!"