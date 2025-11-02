#!/usr/bin/env python3
"""
TEST SPECIFIC IMAGE WITH HYBRID MODEL
====        print("\n🤖 Model Components:")
        if 'yolo_result' in result:
            yolo_conf = result['yolo_result'].get('confidence', 'N/A')
            if isinstance(yolo_conf, (int, float)):
                print(f"   YOLO v11: {yolo_conf:.2f}")
            else:
                print(f"   YOLO v11: {yolo_conf}")
        if 'fracnet_result' in result:
            fracnet_conf = result['fracnet_result'].get('confidence', 'N/A')
            if isinstance(fracnet_conf, (int, float)):
                print(f"   FracNet2D: {fracnet_conf:.2f}")
            else:
                print(f"   FracNet2D: {fracnet_conf}")==============================

Simple script to test the hybrid fracture detection model on a specific image.

Usage:
    python test_specific_image.py "path/to/your/image.jpg"

Or modify the image_path variable below to test different images.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from final_hybrid_model import FinalHybridFractureDetector

def test_specific_image(image_path: str):
    """
    Test the hybrid model on a specific image file.

    Args:
        image_path: Path to the image file to test
    """
    print("🩺 Testing Specific Image with Hybrid Model")
    print("=" * 50)

    # Validate image path
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return False

    # Check if it's an image file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"❌ Error: File is not a valid image format: {image_path}")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        return False

    try:
        # Initialize the hybrid model
        print("🚀 Initializing Final Hybrid Fracture Detection System...")
        detector = FinalHybridFractureDetector()

        # Make prediction
        print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
        result = detector.predict(image_path)

        # Display results
        print("\n" + "=" * 50)
        print("🏆 PREDICTION RESULTS")
        print("=" * 50)

        prediction = result['is_fracture']
        confidence = result['confidence']
        decision_reason = result['decision_reason']

        # Format prediction
        pred_text = "FRACTURE DETECTED" if prediction == 1 else "NO FRACTURE"
        pred_emoji = "🦴" if prediction == 1 else "✅"

        print(f"{pred_emoji} **{pred_text}**")
        print(f"📊 Confidence: {confidence:.2f}")
        print(f"🎯 Decision Strategy: {decision_reason}")

        # Model details
        print("\n🤖 Model Components:")
        if 'yolo_result' in result:
            yolo_conf = result['yolo_result'].get('confidence', 'N/A')
            if isinstance(yolo_conf, (int, float)):
                print(f"   YOLO v11: {yolo_conf:.2f}")
            else:
                print(f"   YOLO v11: {yolo_conf}")
        if 'fracnet_result' in result:
            fracnet_conf = result['fracnet_result'].get('confidence', 'N/A')
            if isinstance(fracnet_conf, (int, float)):
                print(f"   FracNet2D: {fracnet_conf:.2f}")
            else:
                print(f"   FracNet2D: {fracnet_conf}")

        # Additional metrics
        if 'yolo_confidence' in result:
            print(f"🔍 YOLO Confidence: {result['yolo_confidence']:.2f}")
        if 'fracnet_confidence' in result:
            print(f"🔬 FracNet Confidence: {result['fracnet_confidence']:.2f}")

        print("\n📊 Decision Breakdown:")
        if 'decision_reasons' in result:
            for reason, count in result['decision_reasons'].items():
                print(f"   {reason}: {count}")
                print(f"   {reason}: {count}")

        print("\n" + "=" * 50)
        print("✅ Analysis Complete!")
        return True

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Test hybrid fracture detection model on a specific image')
    parser.add_argument('image_path', nargs='?', help='Path to the image file to test')
    parser.add_argument('--list-examples', action='store_true', help='List example images from the dataset')

    args = parser.parse_args()

    if args.list_examples:
        print("📋 Available test images from dataset:")
        test_fracture_dir = Path("dataset/binary_classification/test/fracture")
        test_no_fracture_dir = Path("dataset/binary_classification/test/no_fracture")

        if test_fracture_dir.exists():
            fracture_images = list(test_fracture_dir.glob("*.jpg"))[:5]  # Show first 5
            print("🦴 Fracture examples:")
            for img in fracture_images:
                print(f"   {img}")

        if test_no_fracture_dir.exists():
            no_fracture_images = list(test_no_fracture_dir.glob("*.jpg"))[:5]  # Show first 5
            print("✅ No fracture examples:")
            for img in no_fracture_images:
                print(f"   {img}")
        return

    # Use provided path or default example
    if args.image_path:
        image_path = args.image_path
    else:
        # Default example image
        image_path = r"dataset\binary_classification\test\fracture\118_jpg.rf.8cd4df10107a0fc9af4d43db91dae469.jpg"
        print(f"ℹ️  No image path provided, using example: {image_path}")
        print("💡 Tip: Use --list-examples to see available test images")
        print("💡 Tip: Or provide image path as argument: python test_specific_image.py \"path/to/image.jpg\"")
        print()

    # Run the test
    success = test_specific_image(image_path)

    if not success:
        print("\n❌ Test failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()