import cv2
import numpy as np
import sys

# Load video
input_video_path = "./inputvid/trimmed_output_video.mp4"  # Input video path
output_video_path = "./outvid/output_fourthshot.mp4"  # Output video path

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Get original video dimensions and total frame count
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object with the same frame size as the original
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

# Define the parameters for cylindrical mapping
def cylindrical_warp(img):
    """Warp the input image to a cylindrical perspective."""
    h, w = img.shape[:2]
    f = w / 3  # Focal length, adjust for flattening amount

    # Generate the cylindrical warp transformation
    cylinder = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            # Mapping each pixel
            theta = (x - w / 2) / f
            h_ = (y - h / 2) / f
            X = int(f * np.tan(theta) + w / 2)
            Y = int(f * h_ / np.cos(theta) + h / 2)
            if 0 <= X < w and 0 <= Y < h:
                cylinder[Y, X] = img[y, x]
    return cylinder

# Process each frame with a progress tracker
for frame_index in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flatten the frame
    flattened_frame = cylindrical_warp(frame)
    
    # Write the frame to the output video
    out.write(flattened_frame)

    # Print progress
    progress = (frame_index + 1) / total_frames * 100
    print(f"Progress: {progress:.2f}% ({frame_index + 1}/{total_frames} frames processed)", end='\r')

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print("\nVideo processing complete!")
