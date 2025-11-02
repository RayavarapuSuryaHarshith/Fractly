"""
HYBRID MODEL VISUAL FLOW DIAGRAM
================================

                    📷 INPUT X-RAY IMAGE
                           │
                           ▼
                 ┌─────────────────────┐
                 │   PREPROCESSING     │
                 │ ─────────────────── │
                 │ • RGB Conversion    │
                 │ • Enhancement       │
                 │ • Noise Reduction   │
                 │ • Normalization     │
                 └─────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
   ┌─────────────────────┐    ┌─────────────────────┐
   │    YOLO v11         │    │   CUSTOM FRACNET   │
   │   (Detection)       │    │  (Classification)   │
   │ ─────────────────── │    │ ─────────────────── │
   │ Input: 640×640      │    │ Input: 224×224      │
   │ Output: Bboxes      │    │ Output: Binary      │
   │ Classes: 4          │    │ Classes: 2          │
   │ • angle             │    │ • fracture          │
   │ • fracture          │    │ • no_fracture       │
   │ • line              │    │                     │
   │ • messed_up_angle   │    │ Features:           │
   │                     │    │ • Uncertainty       │
   │ Confidence: 0-1     │    │ • Temperature       │
   └─────────────────────┘    │ • Multi-scale TTA   │
              │                └─────────────────────┘
              ▼                          │
   ┌─────────────────────┐              ▼
   │  YOLO PROCESSING    │    ┌─────────────────────┐
   │ ─────────────────── │    │ FRACNET PROCESSING  │
   │ • Parse detections  │    │ ─────────────────── │
   │ • Extract fracture  │    │ • Uncertainty calc  │
   │ • Confidence calc   │    │ • Temperature scale │
   │ • Indirect signals  │    │ • Threshold adjust  │
   └─────────────────────┘    └─────────────────────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
                 ┌─────────────────────┐
                 │  ENSEMBLE FUSION    │
                 │ ─────────────────── │
                 │                     │
                 │ LEVEL 1: High Conf  │
                 │ ├─ Both > threshold │
                 │ └─ Direct decision  │
                 │                     │
                 │ LEVEL 2: Consensus  │
                 │ ├─ Models agree     │
                 │ └─ Bonus weighting  │
                 │                     │
                 │ LEVEL 3: Conflict   │
                 │ ├─ FracNet favored  │
                 │ └─ Medical priority │
                 │                     │
                 │ LEVEL 4: Uncertain  │
                 │ ├─ Conservative     │
                 │ └─ Safety first     │
                 └─────────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │  FINAL DECISION     │
                 │ ─────────────────── │
                 │ • is_fracture: bool │
                 │ • confidence: float │
                 │ • decision_method   │
                 │ • component_scores  │
                 │ • uncertainty       │
                 └─────────────────────┘
                           │
                           ▼
                    📊 STRUCTURED OUTPUT

WEIGHT DISTRIBUTION:
===================

Default Weights:
├─ YOLO Weight: 0.30 (reduced due to current poor performance)
├─ FracNet Weight: 0.70 (increased for medical accuracy)
├─ Consensus Bonus: 0.25 (when both models agree)
├─ High Confidence Bonus: 0.15 (very confident predictions)
└─ Consistency Bonus: 0.10 (consistent across augmentations)

Thresholds:
├─ YOLO Threshold: 0.20 (lowered for better sensitivity)
├─ FracNet Threshold: 0.35 (balanced for medical use)
├─ Consensus Threshold: 0.30 (ensemble decision point)
└─ High Confidence: 0.75 (very confident decisions)

DECISION FLOW:
=============

1. IF (YOLO_conf > 0.7 AND FracNet_conf > 0.7):
   → FRACTURE with high confidence

2. ELIF (YOLO_conf < 0.2 AND FracNet_conf < 0.3):
   → NO FRACTURE with high confidence

3. ELIF (YOLO_agrees AND FracNet_agrees):
   → Consensus decision with bonus

4. ELIF (disagreement):
   → FracNet-weighted decision (medical priority)

5. ELSE:
   → Conservative decision with uncertainty penalty

PERFORMANCE MONITORING:
======================

Current Issues:
├─ YOLO: Not detecting objects (0.000 confidence)
├─ Reason: Object detection model vs classification expectation
├─ Impact: Hybrid model defaults to "NO FRACTURE"
└─ Solution: Retrain YOLO or fix detection interpretation

Expected After Fixes:
├─ Overall Accuracy: 90%+
├─ Fracture Detection: 85%+
├─ False Positive Rate: <5%
└─ Clinical Reliability: High
"""