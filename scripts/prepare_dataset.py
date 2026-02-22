#!/usr/bin/env python3
"""
Script to convert the provided dataset format to the expected CSV format.
Input:  FEN_STRING Label
Output: FEN_STRING;Label
"""

import os
import sys

def convert_file(input_path, output_path):
    """Convert a single dataset file."""
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            # Split on last space to get label
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            fen, label = parts
            # Write in CSV format
            outfile.write(f"{fen};{label}\n")

def main():
    dataset_dir = "dataset"
    output_dir = "dataset_converted"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each category
    for category in ["check", "checkmate", "nothing"]:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            continue
        
        for filename in os.listdir(cat_dir):
            if not filename.endswith('.txt'):
                continue
            
            input_path = os.path.join(cat_dir, filename)
            output_filename = f"{category}_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Converting {input_path} -> {output_path}")
            convert_file(input_path, output_path)
    
    # Merge all files into one
    merged_path = os.path.join(output_dir, "merged_dataset.csv")
    print(f"\nMerging all files into {merged_path}")
    
    with open(merged_path, 'w') as merged:
        for filename in sorted(os.listdir(output_dir)):
            if filename == "merged_dataset.csv":
                continue
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                merged.write(f.read())
    
    print(f"\nDone! Merged dataset: {merged_path}")
    print(f"Total lines: {sum(1 for _ in open(merged_path))}")

if __name__ == "__main__":
    main()
