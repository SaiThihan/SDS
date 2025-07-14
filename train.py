from ultralytics import YOLO
import torch

def main():
    # Load YOLOv8 segmentation model (tiny version for speed)
    model = YOLO('yolov8n-seg.pt')  # Use the segmentation variant
    
    # Determine device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Disable AMP for GTX 1650 to prevent NaN issues
    amp_enabled = False if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1650" else True

    # Train the model
    results = model.train(
        data='data/data.yaml',  # Ensure this path is correct
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=4,
        amp=amp_enabled,  # Fix: Disable AMP for GTX 1650
        save=True,  # Ensure model is saved
        project="runs/segment",  # Explicitly set project path for segmentation
        name="train",  # Ensure it saves under train/
    )

if __name__ == "__main__":
    main()
