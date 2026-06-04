"""Fine-tune YOLOv8 to detect emergency vehicles (ambulance, fire truck, police).

Usage:
    python src/train_emergency_detector.py --api-key YOUR_ROBOFLOW_API_KEY

Steps:
    1. Creates a free account at https://roboflow.com
    2. Go to https://app.roboflow.com and get your API key from Settings -> Roboflow API
    3. Run this script with your key
    4. The trained model is saved to results/emergency_detector/best.pt
    5. Use it with:
       python src/main.py detect-live --source 0 --model-name results/emergency_detector/best.pt
       python src/main.py live --checkpoint ... --emergency-model results/emergency_detector/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for emergency vehicle detection")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (50 is enough for a good result)")
    parser.add_argument("--output-dir", default="results/emergency_detector")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: download the dataset ---
    print("Downloading emergency vehicle dataset from Roboflow...")
    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    # Public emergency vehicle dataset — ambulance, fire truck, police car
    # https://universe.roboflow.com/emergency-vehicle-detection/emergency-vehicles-mxvlq
    project = rf.workspace("emergency-vehicle-detection").project("emergency-vehicles-mxvlq")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(output_dir / "dataset"))
    data_yaml = str(Path(dataset.location) / "data.yaml")
    print(f"Dataset downloaded to: {dataset.location}")
    print(f"Classes: {dataset.classes}")

    # --- Step 2: fine-tune ---
    print(f"\nFine-tuning YOLOv8n for {args.epochs} epochs...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=15,
        batch=16,
        lr0=0.01,
        cos_lr=True,
    )

    # --- Step 3: copy best weights to a predictable location ---
    best_weights = Path(output_dir) / "train" / "weights" / "best.pt"
    final_path = output_dir / "best.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, final_path)
        print(f"\nTraining complete.")
        print(f"Best model saved to: {final_path}")
        print(f"\nTo use it for live detection:")
        print(f"  python src/main.py detect-live --source 0 --model-name {final_path}")
        print(f"\nTo use it for emergency preemption:")
        print(f"  python src/main.py live --checkpoint results/lagos_peak_rl/best_dqn_model.pt --emergency-model {final_path}")
    else:
        print(f"Training complete. Check {output_dir / 'train'} for weights.")

    # --- Step 4: quick validation ---
    print("\nRunning validation on test set...")
    best_model = YOLO(str(final_path))
    metrics = best_model.val(data=data_yaml)
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
