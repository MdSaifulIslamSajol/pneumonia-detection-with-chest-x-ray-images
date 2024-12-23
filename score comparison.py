import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data preparation
data = {
    "Model": [
        "Swinv2", "ViT", "ConvNextV2", "Cvt", "EfficientFormer", "PvtV2",
        "MobileViTV2", "ResNet50", "VGG16", "MobileNet", "GoogleNet", "EfficientNetB0"
    ],
    "Test Accuracy": [95.03, 94.71, 93.91, 94.71, 93.75, 94.39, 92.79, 94.55, 93.75, 94.23, 94.39, 94.71],
    "Precision": [0.9474, 0.9470, 0.9379, 0.9433, 0.9374, 0.9399, 0.9306, 0.9496, 0.9413, 0.9373, 0.9419, 0.9519],
    "Recall": [0.9466, 0.9397, 0.9316, 0.9440, 0.9286, 0.9406, 0.9150, 0.9342, 0.9252, 0.9402, 0.9380, 0.9355],
    "F1 Score": [0.9470, 0.9431, 0.9346, 0.9436, 0.9327, 0.9402, 0.9217, 0.9409, 0.9322, 0.9387, 0.9399, 0.9426],
    "Class 0 Accuracy": [93.16, 91.03, 90.17, 93.16, 89.32, 92.74, 86.32, 88.89, 87.61, 93.16, 91.45, 88.89],
    "Class 1 Accuracy": [96.15, 96.92, 96.15, 95.64, 96.41, 95.38, 96.67, 97.95, 97.44, 94.87, 96.15, 98.21]
}

df = pd.DataFrame(data)

# Define colors for each model
colors = plt.cm.get_cmap('tab20', len(df['Model']))

# Plotting
plt.figure(figsize=(16, 12))

# Test Accuracy
plt.subplot(2, 2, 1)
plt.bar(df['Model'], df['Test Accuracy'], color=colors(np.arange(len(df['Model']))))
plt.ylim(80, 101)
plt.title('Test Accuracy')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')

# F1 Score
plt.subplot(2, 2, 2)
plt.bar(df['Model'], df['F1 Score'], color=colors(np.arange(len(df['Model']))))
plt.ylim(0.85, 1.01)
plt.title('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('F1 Score')

# Precision
plt.subplot(2, 2, 3)
plt.bar(df['Model'], df['Precision'], color=colors(np.arange(len(df['Model']))))
plt.ylim(0.85, 1.01)
plt.title('Precision')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('Precision')

# Recall
plt.subplot(2, 2, 4)
plt.bar(df['Model'], df['Recall'], color=colors(np.arange(len(df['Model']))))
plt.ylim(0.85, 1.01)
plt.title('Recall')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('Recall')

plt.tight_layout()
plt.savefig(f"combined_precision_recall_f1_curve.png", dpi=400)  # Save the figure with high resolution
plt.show()


plt.figure(figsize=(16, 6))

# Class 0 Accuracy
plt.subplot(1, 2, 1)
plt.bar(df['Model'], df['Class 0 Accuracy'], color=colors(np.arange(len(df['Model']))))
plt.ylim(85, 101)
plt.title('Class 0 Accuracy')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')

# Class 1 Accuracy
plt.subplot(1, 2, 2)
plt.bar(df['Model'], df['Class 1 Accuracy'], color=colors(np.arange(len(df['Model']))))
plt.ylim(85, 101)
plt.title('Class 1 Accuracy')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(f"combined_classwise_accuracy_curve.png", dpi=400)  # Save the figure with high resolution

plt.show()
