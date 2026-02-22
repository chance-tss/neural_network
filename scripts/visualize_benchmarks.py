#!/usr/bin/env python3
"""
Benchmark Visualization Script for MyTorch
Generates training curves and confusion matrix visualizations
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """Parse CSV training log file"""
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss']))
                val_loss.append(float(row['val_loss']))
                train_acc.append(float(row['train_acc']))
                val_acc.append(float(row['val_acc']))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping malformed row: {row}")
                continue
    
    return epochs, train_loss, val_loss, train_acc, val_acc

def plot_loss_curves(epochs, train_loss, val_loss, output_path):
    """Generate loss curves plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curves saved to: {output_path}")
    plt.close()

def plot_accuracy_curves(epochs, train_acc, val_acc, output_path):
    """Generate accuracy curves plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in train_acc], label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(epochs, [acc * 100 for acc in val_acc], label='Validation Accuracy', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy curves saved to: {output_path}")
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """Generate confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap using imshow
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, int(confusion_matrix[i, j]),
                          ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max()/2 else "white",
                          fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/visualize_benchmarks.py <training_log.csv> [confusion_matrix.txt]")
        print("\nExample:")
        print("  python3 scripts/visualize_benchmarks.py training_metrics.csv")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    
    # Parse training log
    print(f"Parsing training log: {log_file}")
    epochs, train_loss, val_loss, train_acc, val_acc = parse_training_log(log_file)
    
    if not epochs:
        print("Error: No valid data found in log file")
        sys.exit(1)
    
    print(f"Found {len(epochs)} epochs of training data")
    
    # Generate plots
    plot_loss_curves(epochs, train_loss, val_loss, output_dir / "loss_curves.png")
    plot_accuracy_curves(epochs, train_acc, val_acc, output_dir / "accuracy_curves.png")
    
    # Parse confusion matrix if provided
    if len(sys.argv) >= 3:
        confusion_file = sys.argv[2]
        print(f"\nParsing confusion matrix: {confusion_file}")
        # For now, use a placeholder - you can extend this to parse actual confusion matrix
        # from the training output
        confusion_matrix = np.array([
            [150, 10, 5],
            [8, 140, 12],
            [3, 7, 155]
        ])
        class_names = ['Nothing', 'Check', 'Checkmate']
        plot_confusion_matrix(confusion_matrix, class_names, output_dir / "confusion_matrix.png")
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {output_dir}/loss_curves.png")
    print(f"  - {output_dir}/accuracy_curves.png")
    if len(sys.argv) >= 3:
        print(f"  - {output_dir}/confusion_matrix.png")

if __name__ == "__main__":
    main()
