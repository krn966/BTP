import cv2
import numpy as np
import sys

# File paths
input_video_path = "./outvid/final.mp4"  # Input video path
output_video_path = "./outvid/final2.mp4"  # Output video path

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

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

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

    # Detect gridlines in the current frame by identifying edges
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the mask for gap areas by thresholding the difference
    diff = cv2.absdiff(gray_prev, gray_next)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])

    # Interpolate using the average of adjacent frames
    interpolated_frame = cv2.addWeighted(prev_frame, 0.5, next_frame, 0.5, 0)

    # Use the mask to combine the original and interpolated frames
    final_frame = cv2.bitwise_and(prev_frame, cv2.bitwise_not(mask_3ch)) + cv2.bitwise_and(interpolated_frame, mask_3ch)
    
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
