from ultralytics import YOLO
import cv2

def main():
    # Path to the trained model weights (adjust if needed)
    model_path = 'runs/segment/train/weights/best.pt'

    try:
        # Load trained model
        model = YOLO(model_path)  # Ensure path to best.pt is correct
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please check the path.")
        return

    # Open the webcam for inference
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Perform inference on the frame
        results = model(frame)  # Pass the frame directly for prediction
        
        # Extract the predicted image with annotations
        annotated_frame = results[0].plot()  # Use the plot() method to draw predictions on the frame

        # Show the annotated frame
        cv2.imshow("Webcam Detection", annotated_frame)

        # Exit condition (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
