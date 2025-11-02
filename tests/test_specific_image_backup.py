#!/usr/bin/env python3
"""
TEST SPECIFIC IMAGE
===================

Test the hybrid model on a specific image file.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_specific_image():
    # Image path - CHANGE THIS TO TEST DIFFERENT IMAGES
    image_path = r"dataset\dataset\test\images\202_jpg.rf.c40f90a31d2a706615376be9ec1f6002.jpg"
    label_path = r"dataset\dataset\test\labels\202_jpg.rf.c40f90a31d2a706615376be9ec1f6002.txt"

    print("🩺 Testing Specific Image with Hybrid Model")
    print("=" * 50)
    print(f"📁 Image: {image_path}")
    print(f"🏷️  Label: {label_path}")
    print()

    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False

    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return False

    print("✅ Files exist")

    # Get ground truth from label file
    print("🎯 Reading ground truth...")
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()

        if content:
            lines = content.split('\n')
            classes = [int(line.split()[0]) for line in lines]
            unique_classes = set(classes)

            print(f"📄 Label file: {len(lines)} objects, classes: {sorted(unique_classes)}")

            # Interpret classes (0=no_fracture, 1=fracture in YOLO format)
            if 0 in unique_classes:
                ground_truth = "NO FRACTURE"
                print("🎯 Ground truth: NO FRACTURE (class 0 detected)")
            elif 1 in unique_classes:
                ground_truth = "FRACTURE"
                print("🎯 Ground truth: FRACTURE (class 1 detected)")
            else:
                ground_truth = "UNKNOWN"
                print("❓ Ground truth: UNKNOWN (unexpected classes)")
        else:
            ground_truth = "UNKNOWN"
            print("❌ Label file is empty")
    except Exception as e:
        print(f"❌ Error reading label file: {e}")
        ground_truth = "UNKNOWN"

    try:
        # Import and initialize model
        from src.final_hybrid_model import FinalHybridFractureDetector

        print("🔧 Loading hybrid model...")
        model = FinalHybridFractureDetector(
            yolo_weights_path="weights/yolo_binary_high_accuracy.pt",
            fracnet_weights_path="weights/custom_fracnet2d_trained_5_epochs.pth",
            device="cpu",
            verbose=False
        )
        print("✅ Model loaded successfully")

        # Make prediction
        print("🔍 Analyzing image...")
        result = model.predict(image_path)

        # Display results
        print("\n" + "=" * 50)
        print("📋 PREDICTION RESULTS")
        print("=" * 50)

        # Main result
        prediction = "FRACTURE" if result['is_fracture'] else "NO FRACTURE"

        if result['is_fracture']:
            print("🚨 DIAGNOSIS: FRACTURE DETECTED")
        else:
            print("✅ DIAGNOSIS: NO FRACTURE DETECTED")

        print(".1f")
        print(f"🎯 Ground Truth: {ground_truth}")

        # Compare prediction vs ground truth
        if prediction == ground_truth:
            print("✅ RESULT: CORRECT CLASSIFICATION! 🎉")
            status = "CORRECT"
        else:
            print("❌ RESULT: MISCLASSIFICATION! ⚠️")
            status = "INCORRECT"

        print(f"📊 Status: {status}")
        print(f"🤝 Models Agree: {'✅ Yes' if result['models_agree'] else '⚠️ No'}")
        print(f"📋 Decision Logic: {result['decision_reason']}")

        print("\n" + "-" * 30)
        print("📊 DETAILED METRICS")
        print("-" * 30)

        # Individual model results
        yolo_result = result['yolo_result']
        fracnet_result = result['fracnet_result']

        print(".1%")
        print(".1%")

        print(f"YOLO Prediction: {'Fracture' if yolo_result['is_fracture'] else 'No Fracture'}")
        print(f"FracNet Prediction: {'Fracture' if fracnet_result['is_fracture'] else 'No Fracture'}")

        print("\n" + "=" * 50)
        print("✅ Analysis Complete!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_specific_image()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")