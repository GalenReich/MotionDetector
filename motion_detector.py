# Motion detector using OpenCV
# This program will detect motion in a series of video frames
# The videos are stored in the ./videos directory
# For efficiency, the program will only process every few seconds of video

import cv2
import numpy as np
import os
import multiprocessing as mp
import tqdm

# Crop the video to the region of interest which is the lower portion of the video
def crop_frame(frame):
    height, width = frame.shape[:2]
    start_row, start_col = int(height * 0.50), 0 # From halfway down the video
    end_row, end_col = height, width
    cropped_frame = frame[start_row:end_row, start_col:end_col]
    return cropped_frame

# Define a function to process the video file
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    # Use motion detection to process every 3 seconds of video
    interval = 3
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    # Set motion detection parameters
    threshold = 50000
    blur_amount = 5

    # Load the mask image
    mask = cv2.imread('mask.jpg', 0)

    # Open the video file and process the frames, load pairs of frames and check for motion
    while cap.isOpened():
        ret1, frame1 = cap.read()
        if ret1: frame_count += 1
        if frame1 is None: break
        # Check if next frame is available
        if not cap.isOpened(): break
        ret2, frame2 = cap.read()
        if frame2 is None: break
        if ret2: frame_count += 1
        if ret1 and ret2:
            if frame_count % (interval * frame_rate) == 0:
                # Process both frames
                frame1 = crop_frame(frame1)
                frame2 = crop_frame(frame2)
                # Convert the frames to grayscale
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                # Apply the mask to the frames
                gray1 = cv2.bitwise_and(gray1, mask)
                gray2 = cv2.bitwise_and(gray2, mask)
                # Apply Median blur to the frames
                gray1 = cv2.medianBlur(gray1, blur_amount)
                gray2 = cv2.medianBlur(gray2, blur_amount)
                # Calculate the difference between the two frames
                delta = cv2.absdiff(gray1, gray2)
                # Sum the differences to get a single value
                delta_sum = np.sum(delta)

                # If the difference is significant, save the frame to the ./output directory   
                if delta_sum > threshold:
                    # Use the original video name and the frame number as the filename
                    filename = f'output_all_images/{os.path.basename(video_file)}_{frame_count}_{delta_sum}.jpg'
                    cv2.imwrite(filename, frame1)
                    #print(f'Car detected in frame {frame_count-1}')
                else:
                    pass
        else:
            pass

if __name__ == '__main__':
    # Load the list of video files, returning full filepath matching mp4 only
    video_directory = './videos'
    video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
    print(f'Found {len(video_files)} video files')

    # # Process the video files using multiple processes
    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(process_video, video_files), total=len(video_files)):
        pass
    pool.close()