from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLO11n model
model = YOLO("yolov11n-face.pt")

# Define path to the image file
source = "ComfyUI_00877_.png_realvisxlV50_v50Bakedvae_0001.jpg"

# Run inference on the source
results = model(source)  # list of Results objects

# Load the original image
img = cv2.imread(source)

# Process results
for r in results:
    # Check if any faces were detected
    if r.boxes is not None:
        # Get the first detected face (you can modify this to handle multiple faces)
        box = r.boxes[0]  # Get the first detection
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        
        # Calculate original width and height
        width = x2 - x1
        height = y2 - y1
        
        # Find the larger dimension
        max_dim = max(width, height)
        
        # Calculate the enlargement needed (we'll use 200px as a base and adjust)
        target_enlargement = 200  # Base enlargement value
        total_size = max_dim + 2 * target_enlargement  # Total size of the square crop
        
        # Calculate center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Calculate new coordinates for square crop
        half_size = total_size // 2
        x1_new = max(0, center_x - half_size)
        y1_new = max(0, center_y - half_size)
        x2_new = min(img.shape[1], center_x + half_size)
        y2_new = min(img.shape[0], center_y + half_size)
        
        # Adjust if we hit image boundaries
        actual_width = x2_new - x1_new
        actual_height = y2_new - y1_new
        final_size = max(actual_width, actual_height)
        
        # Recalculate to ensure square crop
        if actual_width < final_size:
            x1_new = max(0, x2_new - final_size)
        elif actual_height < final_size:
            y1_new = max(0, y2_new - final_size)
        
        x2_new = min(img.shape[1], x1_new + final_size)
        y2_new = min(img.shape[0], y1_new + final_size)
        
        # Crop the face region
        face_crop = img[y1_new:y2_new, x1_new:x2_new]
        
        # Save the cropped face
        cv2.imwrite("face.png", face_crop)
        
        print(f"Face detected and saved to face.png")
        print(f"Original box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"Square crop: x1={x1_new}, y1={y1_new}, x2={x2_new}, y2={y2_new}")
        print(f"Crop size: {final_size}x{final_size}")
    else:
        print("No faces detected in the image")