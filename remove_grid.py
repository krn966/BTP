import cv2
import numpy as np
import sys

# File paths
input_video_path = "./outvid/final.mp4"  # Input video path
output_video_path = "./outvid/final4.mp4"  # Output video path

def remove_gridlines_with_blur(frame):
    """Apply Gaussian blur to reduce the appearance of gridlines."""
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred_frame

# Define the gridline removal function with Fourier applied to each color channel
def remove_gridlines_with_fourier_color(frame):
    """Reduce gridlines by filtering each color channel in the frequency domain."""
    channels = cv2.split(frame)
    processed_channels = []

    for channel in channels:
        # Apply Fourier Transform to each channel
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Mask to filter high frequencies associated with gridlines
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 10  # radius of the mask
        center = (ccol, crow)
        cv2.circle(mask, center, r, (0, 0), -1)

        # Apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize and convert back to uint8
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)
        processed_channels.append(img_back)

    # Merge channels back to color frame
    final_frame = cv2.merge(processed_channels)
    return final_frame

# Define the gridline removal function with selective median filtering
def remove_gridlines_with_selective_median(frame):
    """Reduce gridlines by selectively applying median filtering to maintain quality."""
    # Convert to grayscale and apply edge detection to find gridline regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Apply median filter to the entire frame with a small kernel to reduce blur
    median_filtered = cv2.medianBlur(frame, 3)  # Smaller kernel size for less blurring

    # Use the edges as a mask to combine the original frame and the filtered frame
    mask = cv2.bitwise_not(edges)
    mask_3ch = cv2.merge([mask, mask, mask])  # Convert mask to 3 channels

    # Blend the original and median-filtered frames using the mask
    result = cv2.bitwise_and(frame, mask_3ch) + cv2.bitwise_and(median_filtered, cv2.bitwise_not(mask_3ch))
    
    return result

# Define the gridline removal function with median filtering
def remove_gridlines_with_median(frame):
    """Apply median filtering to reduce gridlines."""
    # Apply median filtering (adjust kernel size if needed)
    median_filtered_frame = cv2.medianBlur(frame, 5)  # Kernel size of 5; increase if gridlines persist
    return median_filtered_frame

# Define the gridline removal function
def remove_gridlines_with_fourier(frame):
    """Reduce gridlines by filtering in the frequency domain."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Fourier Transform
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a mask to filter out high frequencies associated with gridlines
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2

    # Mask parameters (adjust size based on gridline frequency)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 10  # radius of the mask
    center = (ccol, crow)
    cv2.circle(mask, center, r, (0, 0), -1)

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize and convert back to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    # Merge back to three channels to match original frame dimensions
    final_frame = cv2.merge([img_back, img_back, img_back])
    return final_frame

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

# Process each frame
for frame_index in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # Remove gridlines using the Fourier-based function
    processed_frame = remove_gridlines_with_blur(frame)
    
    # Write the processed frame to the output video
    out.write(processed_frame)

    # Print progress
    progress = (frame_index + 1) / total_frames * 100
    print(f"Progress: {progress:.2f}% ({frame_index + 1}/{total_frames} frames processed)", end='\r')

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
print("\nVideo processing complete!")
