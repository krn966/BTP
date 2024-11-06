import cv2
import numpy as np
import sys

# File paths
input_video_path = "./inputvid/trimmed_output_video.mp4"  # Input video path
output_video_path = "./outvid/final.mp4"  # Output video path

# Load video
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define codec and create VideoWriter object with the same frame size as the original
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

# Define the parameters for cylindrical mapping (flattening)
def cylindrical_warp(img):
    """Warp the input image to a cylindrical perspective."""
    h, w = img.shape[:2]
    f = w / 2  # Focal length, adjust for flattening amount

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

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    sys.exit()

# Process each frame, starting from the second frame
for frame_index in range(1, total_frames - 1):
    # Read the next frame
    ret, next_frame = cap.read()
    if not ret:
        break

    # Step 1: Flatten each frame with cylindrical warp
    flattened_prev_frame = cylindrical_warp(prev_frame)
    flattened_next_frame = cylindrical_warp(next_frame)

    # Step 2: Apply gridline blending
    gray_prev = cv2.cvtColor(flattened_prev_frame, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(flattened_next_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the mask for gap areas by thresholding the difference
    diff = cv2.absdiff(gray_prev, gray_next)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])

    # Interpolate using the average of adjacent frames
    interpolated_frame = cv2.addWeighted(flattened_prev_frame, 0.5, flattened_next_frame, 0.5, 0)

    # Use the mask to combine the original and interpolated frames
    final_frame = cv2.bitwise_and(flattened_prev_frame, cv2.bitwise_not(mask_3ch)) + cv2.bitwise_and(interpolated_frame, mask_3ch)
    
    # Write the processed frame to the output video
    out.write(final_frame)

    # Print progress
    progress = (frame_index + 1) / total_frames * 100
    print(f"Progress: {progress:.2f}% ({frame_index + 1}/{total_frames} frames processed)", end='\r')
    
    # Update the previous frame for the next iteration
    prev_frame = next_frame.copy()

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
print("\nVideo processing complete!")
