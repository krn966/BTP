import cv2
import os

def get_video_info(video_path):
    # Check if the video file exists
    if not os.path.isfile(video_path):
        print("Error: Video file does not exist at the specified path.")
        return

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open the video.")
        return

    # Get video properties
    video_info = {
        'Format': int(video_capture.get(cv2.CAP_PROP_FOURCC)),
        'Frame Width': video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        'Frame Height': video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'Frame Rate': video_capture.get(cv2.CAP_PROP_FPS),
        'Total Frames': video_capture.get(cv2.CAP_PROP_FRAME_COUNT),
        'Duration (seconds)': video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / video_capture.get(cv2.CAP_PROP_FPS),
    }

    # Convert FourCC code to a readable format
    fourcc_code = video_info['Format']
    fourcc_string = ''.join([chr((fourcc_code >> (8 * i)) & 0xFF) for i in range(4)])

    # Display video information
    print("Video Information:")
    print(f"Format: {fourcc_string}")
    print(f"Frame Size: {int(video_info['Frame Width'])} x {int(video_info['Frame Height'])}")
    print(f"Frame Rate: {video_info['Frame Rate']:.2f} FPS")
    print(f"Total Frames: {int(video_info['Total Frames'])}")
    print(f"Duration: {video_info['Duration (seconds)']:.2f} seconds")

    # Release the video capture object
    video_capture.release()


# Specify the path to your video file
video_path = './inputvid/trimmed_output_video.mp4'  # Replace with your video file path
get_video_info(video_path)
