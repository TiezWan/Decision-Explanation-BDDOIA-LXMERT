import cv2
import time
import pdb
import numpy as np
import utils


# vidcap = cv2.VideoCapture("00091078-59817bb0.mov")
# timepre=time.time()
# num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(num_frames)
# frames=[]

# for image_num in range(0, int(num_frames)):
#     success, image=vidcap.read()
#     frames.append(image)
    
# timepost=time.time()
# print(timepost-timepre)
# print(len(frames))
# vidcap.release()
# cv2.destroyAllWindows()


utils.video2frames(["00091078-59817bb0.mov"])


# from moviepy.editor import VideoFileClip
# import numpy as np
# import os
# from datetime import timedelta

# def splitinframes(video_file, fps=30):
#     frames=[]
#     # load the video clip
#     video_clip = VideoFileClip(video_file)

#     # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
#     fps = max(min(video_clip.fps, fps),1)
#     # if fps is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
#     step = 1 / fps
#     # iterate over each possible frame
#     for current_timestamp in np.arange(0, video_clip.duration, step):
#         # format the file name and save it
#         # save the frame with the current timestamp
#         frames.append(video_clip.get_frame(current_timestamp))
#     return frames

# currtime=time.time()
# splitinframes("00091078-59817bb0.mov", 30)
# newtime=time.time()
# print(newtime-currtime)