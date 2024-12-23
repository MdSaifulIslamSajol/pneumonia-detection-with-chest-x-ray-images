#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:47:04 2024

@author: saiful
"""

training_accuracy_swinv2 = [
    98.8305, 99.0989, 99.0606, 99.3290, 98.5813, 99.1181, 99.4248,
    99.3290, 99.3673, 99.0222, 99.3865, 99.6549, 99.1564, 99.4248,
    99.5782, 99.6590, 99.3290, 99.5813, 99.6124, 99.7642, 99.4518, 99.7128, 99.8345, 99.4897, 99.8921, 99.8448, 99.8658, 99.7891, 99.7891
]
validation_accuracy_swinv2 = [
    93.4295, 90.7051, 95.0321, 93.7500, 93.5897, 93.5897, 94.5513,
    89.7436, 89.9038, 94.3910, 92.7885, 93.7500, 91.5064, 94.2308,
    94.2308, 89.4231, 94.5501, 90.3245, 91.9468, 92.6121, 93.4095, 91.4587, 93.6012, 94.7295, 90.4395, 90.3245, 92.9487, 93.2692, 93.2692
]
training_accuracy_vit = [
    93.7883, 97.2201, 98.3512, 98.0637, 98.4854, 99.1756, 99.3290,
    98.2745, 99.1564, 99.2906, 99.1564, 99.5782, 99.0606, 99.5974,
    99.4248, 99.0606, 99.5590, 99.7508, 99.6357, 99.5974, 99.3482,
    99.6357, 99.5399, 99.6741, 99.6741, 99.4440, 99.5782, 99.7891,
    99.7891
]
validation_accuracy_vit = [
    90.2244, 88.9423, 92.9487, 87.5000, 85.8974, 90.7051, 81.4103,
    84.4551, 92.4679, 87.5000, 92.6282, 93.7500, 93.2692, 94.0705,
    82.0513, 91.8269, 92.4679, 93.4295, 92.9487, 91.5064, 93.1090,
    91.6667, 86.5385, 91.5064, 92.1474, 94.7115, 91.6667, 91.6667,
    92.1474
]
training_accuracy_convnextv2 = [
    94.2676, 97.5460, 98.6580, 98.4087, 98.2937, 99.3673, 99.4440,
    99.1373, 99.7699, 99.6933, 99.5015, 99.6549, 99.7316, 99.7891,
    99.3290, 99.6549, 99.7316, 99.7699, 99.5590, 99.7124, 99.4824,
    99.8083, 99.3865, 99.8658, 99.9617, 100.0000, 100.0000, 100.0000,
    100.0000
]

training_accuracy_convnextv2 = [
    94.2676, 97.5460, 98.6580, 98.4087, 98.2937, 99.3673, 99.4440,
    99.1373, 99.7699, 99.6933, 99.5015, 99.6549, 99.7316, 99.7891,
    99.3290, 99.6549, 99.7316, 99.7699, 99.5590, 99.7124, 99.4824,
    99.8083, 99.3865, 99.8658, 99.9617, 100.0000, 100.0000, 100.0000,
    100.0000
]
validation_accuracy_convnextv2 = [
    87.0192, 90.5449, 90.7051, 85.4167, 91.0256, 93.9103, 89.7436,
    90.7051, 90.5449, 92.9487, 88.9423, 88.3013, 92.6282, 93.9103,
    93.7500, 93.5897, 89.4231, 93.9103, 93.4295, 93.2692, 93.1090,
    92.7885, 93.4295, 91.3462, 91.9872, 92.7885, 92.6282, 93.1090,
    92.9487
]
training_accuracy_cvt = [
    88.5736, 91.8520, 92.8681, 92.9448, 93.4816, 93.1557, 93.7500,
    93.7692, 93.8267, 94.1334, 94.3827, 93.6158, 94.0567, 94.4402,
    94.4977, 94.4018, 94.5936, 93.9801, 94.4977, 93.9992, 94.7853,
    94.4977, 94.2293, 94.1526, 94.5936, 94.4977, 94.1718, 94.7661,
    94.4018
]
validation_accuracy_cvt = [
    86.5385, 89.5833, 92.6282, 89.5833, 80.1282, 90.0641, 91.5064,
    80.1282, 91.9872, 94.3910, 90.0641, 91.9872, 93.9103, 93.1090,
    92.6282, 92.4679, 87.1795, 92.6282, 93.9103, 92.6282, 92.9487,
    91.9872, 78.0449, 93.4295, 93.5897, 92.6282, 94.7115, 94.0705,
    89.4231
]
training_accuracy_efficientformer = [
    91.9670, 98.0637, 98.4471, 98.7730, 98.8497, 99.1756, 99.1948,
    99.2906, 99.3865, 99.5590, 99.5974, 99.5974, 99.9425, 99.6933,
    99.8083, 99.7891, 99.8658, 99.7891, 99.6933, 99.7316, 99.9041,
    99.8850, 99.8658, 99.6933, 99.8466, 99.8275, 99.8466, 99.8275,
    99.9617
]
validation_accuracy_efficientformer = [
    83.8141, 91.1859, 89.5833, 87.6603, 91.9872, 91.3462, 91.9872,
    86.8590, 86.8590, 93.5897, 91.8269, 87.9808, 92.1474, 93.2692,
    93.7500, 89.9038, 89.4231, 84.1346, 91.1859, 90.5449, 89.7436,
    87.3397, 86.8590, 88.1410, 92.1474, 87.6603, 93.1090, 90.5449,
    91.0256
]
training_accuracy_pvtv2 = [
    92.6764, 97.1051, 97.2968, 98.2745, 98.5621, 99.0031, 99.0798,
    98.9839, 99.1181, 99.3482, 99.5399, 99.3098, 98.9647, 99.4248,
    99.6741, 99.5207, 99.6166, 99.6357, 99.6166, 99.6741, 99.4440,
    99.7891, 99.6166, 99.5974, 99.6933, 99.8083, 99.7124, 99.9041,
    99.6933
]
validation_accuracy_pvtv2 = [
    87.9808, 91.9872, 92.9487, 89.5833, 91.6667, 90.0641, 90.2244,
    91.0256, 93.1090, 90.0641, 89.2628, 91.5064, 87.0192, 93.2692,
    94.0705, 92.7885, 94.3910, 89.9038, 91.3462, 91.5064, 89.5833,
    90.2244, 92.4679, 92.3077, 92.7885, 92.4679, 92.4679, 92.9487,
    93.1090
]
training_accuracy_mobilevitv2 = [
    86.4456, 97.0475, 97.2776, 97.9103, 98.2554, 98.5813, 98.3896,
    98.8497, 99.1373, 99.0414, 99.1181, 99.1948, 99.5590, 99.6741,
    99.4440, 99.3098, 99.5782, 99.6166, 99.7316, 99.7124, 99.6741,
    99.7699, 99.7316, 99.6549, 99.7124, 99.8083, 99.8083, 99.7316,
    99.7508
]
validation_accuracy_mobilevitv2 = [
    87.9808, 87.1795, 79.1667, 86.2179, 87.1795, 87.5000, 86.8590,
    86.5385, 91.5064, 90.3846, 86.6987, 88.9423, 90.0641, 88.9423,
    92.7885, 90.5449, 88.7821, 91.6667, 87.6603, 86.6987, 91.5064,
    90.5449, 91.9872, 91.9872, 89.1026, 90.8654, 87.3397, 90.2244,
    89.9038
]
training_accuracy_resnet50 = [
    95.7439, 98.1979, 98.5429, 99.1948, 98.8497, 99.3098, 99.3290,
    99.4057, 99.5974, 99.5782, 99.6357, 99.6357, 99.6549, 99.7891,
    99.6933, 99.6933, 99.5207, 99.8275, 99.8275, 99.7699, 99.9617,
    99.9425, 99.7891, 99.8083, 99.7891, 99.8466, 99.6933, 99.6549,
    99.6549
]
validation_accuracy_resnet50 = [
    90.5449, 88.6218, 79.6474, 93.1090, 94.0705, 93.9103, 91.3462,
    87.0192, 93.5897, 87.5000, 91.3462, 88.3013, 79.8077, 91.6667,
    88.6218, 91.9872, 92.7885, 92.6282, 92.7885, 86.8590, 92.7885,
    88.9423, 88.4615, 92.6282, 94.5513, 94.0705, 93.1090, 87.5000,
    92.6282
]
training_accuracy_vgg16 = [
    92.1971, 96.6641, 97.6802, 98.4087, 98.1595, 98.8880, 98.5621,
    99.1564, 98.6388, 99.1948, 99.2331, 99.3290, 99.6357, 99.0414,
    99.6933, 99.5590, 99.6357, 99.3865, 99.4248, 99.4824, 99.4248,
    99.6357, 99.7124, 99.7124, 99.2140, 99.5015, 99.5590, 99.8083,
    99.7316
]
validation_accuracy_vgg16 = [
    89.9038, 89.5833, 90.2244, 90.5449, 86.8590, 90.8654, 89.9038,
    90.2244, 91.3462, 87.3397, 90.7051, 92.6282, 83.9744, 90.5449,
    90.7051, 84.7756, 92.6282, 86.2179, 90.8654, 90.5449, 91.9872,
    85.4167, 91.3462, 89.2628, 83.9744, 85.4167, 93.7500, 91.9872,
    92.3077
]
training_accuracy_mobilenet = [
    94.3252, 97.9678, 98.5046, 98.6771, 99.0606, 99.2715, 98.9839,
    99.3673, 99.3673, 99.2906, 99.4632, 99.6933, 99.6741, 99.5590,
    99.7124, 99.5782, 99.6933, 99.7699, 99.8466, 99.6549, 99.6166,
    99.7891, 99.7508, 99.9041, 99.7699, 99.8658, 99.5399, 99.8083,
    99.7508
]
validation_accuracy_mobilenet = [
    91.6667, 86.5385, 89.7436, 87.5000, 89.4231, 88.3013, 87.0192,
    90.8654, 88.3013, 91.5064, 92.9487, 86.6987, 92.1474, 89.7436,
    88.9423, 85.8974, 92.7885, 89.9038, 88.1410, 92.6282, 83.8141,
    90.3846, 93.7500, 90.8654, 91.3462, 94.2308, 87.6603, 91.1859,
    77.7244
]
validation_accuracy_mobilenet = [
    91.6667, 86.5385, 89.7436, 87.5000, 89.4231, 88.3013, 87.0192,
    90.8654, 88.3013, 91.5064, 92.9487, 86.6987, 92.1474, 89.7436,
    88.9423, 85.8974, 92.7885, 89.9038, 88.1410, 92.6282, 83.8141,
    90.3846, 93.7500, 90.8654, 91.3462, 94.2308, 87.6603, 91.1859,
    77.7244
]
training_accuracy_googlenet = [
    94.0184, 97.9486, 98.3129, 98.9072, 99.2331, 99.2523, 99.5015,
    99.7316, 99.5399, 99.7124, 99.5207, 99.5974, 99.7699, 99.6549,
    99.8083, 99.6549, 99.7891, 99.9617, 99.8850, 99.6549, 99.9233,
    99.8466, 99.7891, 99.8658, 99.9041, 99.9617, 99.9233, 99.9617,
    99.9617
]
validation_accuracy_googlenet = [
    81.7308, 90.7051, 90.7051, 94.3910, 90.7051, 94.0705, 90.0641,
    93.9103, 93.1090, 92.4679, 92.7885, 91.9872, 90.2244, 92.4679,
    92.1474, 92.9487, 91.9872, 91.3462, 93.2692, 91.8269, 91.1859,
    92.9487, 89.7436, 93.2692, 93.1090, 94.2308, 91.9872, 94.3910,
    92.1474
]
training_accuracy_efficientnet_b0 = [
    92.1396, 97.3543, 98.1212, 98.4087, 98.7347, 98.8113, 99.1948,
    99.0989, 99.3673, 99.4824, 99.5399, 99.4824, 99.6549, 99.5207,
    99.4824, 99.5015, 99.6549, 99.5590, 99.6933, 99.7316, 99.8275,
    99.5974, 99.8083, 99.7891, 99.8658, 99.7316, 99.7891, 99.7699,
    99.7316
]
validation_accuracy_efficientnet_b0 = [
    87.9808, 89.4231, 92.1474, 92.6282, 92.6282, 92.6282, 91.9872,
    92.3077, 93.2692, 90.2244, 93.2692, 93.4295, 93.1090, 94.2308,
    93.4295, 93.1090, 93.4295, 92.4679, 94.7115, 93.4295, 91.6667,
    94.3910, 94.3910, 88.1410, 94.3910, 92.9487, 91.9872, 92.9487,
    90.0641
]

import matplotlib.pyplot as plt
# Plotting training accuracy
plt.figure(figsize=(14, 10))
# Setting the font sizes for various plot elements
plt.rcParams.update({'font.size': 14})  # Default font size for all text
plt.rcParams['axes.labelsize'] = 16    # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 18    # Font size for the title
plt.rcParams['legend.fontsize'] = 12   # Font size for the legend
plt.plot(training_accuracy_swinv2, label='SwinV2 ')
plt.plot(training_accuracy_vit, label='ViT ')
plt.plot(training_accuracy_convnextv2, label='ConvNeXtV2 ')
plt.plot(training_accuracy_cvt, label='CvT ')
plt.plot(training_accuracy_efficientformer, label='EfficientFormer ')
plt.plot(training_accuracy_pvtv2, label='PVTv2 ')
plt.plot(training_accuracy_mobilevitv2, label='MobileViTv2 ')
plt.plot(training_accuracy_resnet50, label='ResNet50 ')
plt.plot(training_accuracy_vgg16, label='VGG16 ')
plt.plot(training_accuracy_mobilenet, label='MobileNet ')
plt.plot(training_accuracy_googlenet, label='GoogleNet ')
plt.plot(training_accuracy_efficientnet_b0, label='EfficientNet B0 ')

plt.ylim([85, 102])
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.savefig(f"combined_training_accuracy_curve.png", dpi=400)  # Save the figure with high resolution
plt.show()

# Plotting validation accuracy
plt.figure(figsize=(14, 7))
plt.plot(validation_accuracy_swinv2, label='SwinV2 - Validation Accuracy')
plt.plot(validation_accuracy_vit, label='ViT - Validation Accuracy')
plt.plot(validation_accuracy_convnextv2, label='ConvNeXtV2 - Validation Accuracy')
plt.plot(validation_accuracy_cvt, label='CvT - Validation Accuracy')
plt.plot(validation_accuracy_efficientformer, label='EfficientFormer - Validation Accuracy')
plt.plot(validation_accuracy_pvtv2, label='PVTv2 - Validation Accuracy')
plt.plot(validation_accuracy_mobilevitv2, label='MobileViTv2 - Validation Accuracy')
plt.plot(validation_accuracy_resnet50, label='ResNet50 - Validation Accuracy')
plt.plot(validation_accuracy_vgg16, label='VGG16 - Validation Accuracy')
plt.plot(validation_accuracy_mobilenet, label='MobileNet - Validation Accuracy')
plt.plot(validation_accuracy_googlenet, label='GoogleNet - Validation Accuracy')
plt.plot(validation_accuracy_efficientnet_b0, label='EfficientNet B0 - Validation Accuracy')

plt.ylim([80, 110])
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()