import os
from ultralytics import YOLO
import torch

def evaluate(model_path, test_dir):
    # Load the trained model
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        return

    # Get list of test images (ensure the directory exists)
    if not os.path.exists(test_dir):
        print(f"Error: The test directory '{test_dir}' does not exist.")
        return

    test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))]

    if not test_images:
        print(f"No image files found in the test directory '{test_dir}'.")
        return

    # Perform inference on test images
    results = model.predict(source=test_images, conf=0.25)

    # Calculate accuracy (for object detection: counting detection success)
    total = len(test_images)
    correct = 0

    for result in results:
        if result.boxes.cls.numel() > 0:  # Check if there are any detected objects
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")
    
    # Optionally, you can print out more detailed results
    print(f"Total images: {total}, Correctly detected: {correct}")

if __name__ == "__main__":
    # Set the path to the test dataset and the model weights
    test_dir = 'data/images/test'  # Path to test images
    model_path = 'runs/segment/train/weights/best.pt' 
    
    # Call the evaluate function
    evaluate(model_path, test_dir)
