"""
Train a rust/corrosion detector.

USAGE:
    1. Download a YOLOv11 dataset from Roboflow.
    2. Unzip it into a folder called 'dataset' next to this script.
       After unzipping you should have: dataset/data.yaml, dataset/train/, dataset/valid/
    3. Run:  python train.py

When training finishes, your trained model will be at:
    runs_v3/detect/runs_v3/detect/trainv3_s/weights/best.pt
"""

from ultralytics import YOLO
import torch
import os
import sys

if __name__ == '__main__':
    # --- sanity checks ---
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will run on CPU and be very slow.")
        print("Install GPU PyTorch with:")
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
        device = "cpu"
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        device = 0

    DATA_YAML = "datasets/v3_big/data.yaml"
    if not os.path.exists(DATA_YAML):
        print(f"\nERROR: '{DATA_YAML}' not found.")
        print("Download a YOLOv11 dataset from Roboflow Universe and unzip it")
        print("into a folder named 'datasets/v3_big' next to this script.")
        sys.exit(1)

    # --- train ---
    model = YOLO("yolo11s.pt")   # 's' = small; 'yolo11n.pt' (nano) is faster but less accurate

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,            # if you get an out-of-memory error, lower this to 8 or 4
        device=device,
        patience=30,         # stop early if no improvement
        project="runs_v3/detect",
        name="trainv3_s",
    )

    print("\nDone. Trained model saved to: runs_v3/detect/runs_v3/detect/trainv3_s/weights/best.pt")
