# Model Weights Directory

This directory contains the trained model weights for the Hybrid Fracture Detection System.

## Required Files

Due to file size limitations, model weights are not included in the repository. You need the following files:

### YOLO v11 Weights
- `yolo11n_fracture_high_accuracy.pt` - Primary YOLO model (88.6% accuracy)
- Alternative: `yolo11n_fracture_trained.pt`

### FracNet2D Weights
- `custom_fracnet2d_best.pth` - Custom FracNet model (56.63% accuracy)

## Download Instructions

1. **Contact the Development Team** for access to trained model weights
2. Place the downloaded `.pt` and `.pth` files in this directory
3. Ensure file names match exactly as listed above

## Model Information

- **Hybrid System Accuracy**: 93.18%
- **YOLO v11 Size**: ~6-10 MB
- **FracNet2D Size**: ~50-100 MB
- **Combined System**: Achieves medical-grade performance

## File Structure

```
weights/
├── README.md (this file)
├── yolo11n_fracture_high_accuracy.pt (required)
├── custom_fracnet2d_best.pth (required)
└── production/ (optional - production models)
```

## Notes

- Model weights are excluded from version control due to large file sizes
- Use Git LFS or cloud storage for sharing model files
- Always verify model checksums before deployment

---

**Development Team:**
- R.Surya Harshith - 22BCE9912
- S.Narendar Reddy - 22BCE7427
- R.Sasi Varshith - 22BCE20052
