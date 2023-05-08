from dotenv import load_dotenv
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imp import load_module
from model import get_model


def detect_and_plot(image, model):
    # Detect the objects in the image
    results = model.detect([image], verbose=0)

    # Get the first result
    r = results[0]


    # Plot the image with the bounding boxes
    plot(image, rect=r['rois'][0])

model = get_model()

def video_to_frames(video, framecount = 100):

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame number
    frame_number = 0

    # Loop through each frame in the video
    while video.isOpened():
        # Read the next frame
        ret, frame = video.read()

        # Check if frame was read successfully
        if ret:
            # Save the frame as a JPEG image
            filename = f"frame_{frame_number:06d}.jpg"
            cv2.imwrite('./images/' + filename, frame)

            # Increment the frame number
            frame_number += 1
        else:
            break

def plot(frame, rect = None):
    #x, y, w, h = 150, 50, 200, 200
    if rect is not None:
        rect = np.array(rect, dtype=np.int32)
        color = (0, 0, 255)
        thickness = 2
        #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        cv2.polylines(frame, [rect], True, (0, 0, 255), 2)
    
    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Plot the frame using matplotlib
    plt.imshow(frame)
    plt.axis("off")
    plt.show()

def getFrame(seconds, fps, video_url):
    cap = cv2.VideoCapture(video_url)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seconds * fps)
    ret, frame = cap.read()
    return frame