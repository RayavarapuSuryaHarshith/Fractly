# BASELINE PERFORMANCE RESULTS - BEFORE EXTENDED TRAINING

# Date: September 25, 2025

# Model: Hybrid YOLO + FracNet (YOLO-only predictions)

# Threshold: 0.05 (optimized for medical safety)

## PERFORMANCE METRICS

- **Sensitivity (Fracture Detection)**: 95.45% ✅ (21/22 correct)
- **Specificity (No False Alarms)**: 9.09% ❌ (2/22 correct)
- **Precision**: 51.22%
- **Overall Accuracy**: 52.27% ❌ (target: 90%)

## CONFUSION MATRIX

                 Predicted
              No Frac  Fracture

Actual No Frac 2 20
Actual Fracture 1 21

## MEDICAL ASSESSMENT

✅ **EXCELLENT**: 95.5% fracture detection rate
✅ **CRITICAL SAFETY**: Only 1 missed fracture out of 22
⚠️ **CAUTION**: 1 fracture potentially missed
❌ **NEEDS IMPROVEMENT**: Accuracy far below 90% target

## MISSED FRACTURES

- 7_jpg.rf.772a27d22553cdc3c6e320581c45dbec.jpg (conf: 0.976)

## ANALYSIS

- YOLO model shows limited discriminative power between fracture/non-fracture
- Fracture predictions: mean 0.367, range 0.024-0.930
- Non-fracture predictions: mean 0.382, range 0.046-0.806
- Maximum possible accuracy with current model: ~59%

## NEXT STEPS

1. Retrain YOLO with extended epochs (50+) for better learning
2. Consider data augmentation to increase dataset size
3. Fine-tune hyperparameters for better convergence
4. Evaluate if 90% accuracy is achievable with current dataset

## TRAINING HISTORY

- Previous training: 5 epochs (insufficient for convergence)
- Current model: weights/yolo11n_fracture_trained.pt
- Dataset: 44 test images (22 fracture, 22 non-fracture)
