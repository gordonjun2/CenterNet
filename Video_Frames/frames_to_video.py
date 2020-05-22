import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    pathIn= '/home/coffeemix/Desktop/Gordon/CenterTrack/data/kitti_tracking/data_tracking_image_2/testing/image_02/0009/'
    pathOut = '/home/coffeemix/Desktop/kitti_test_0009.mp4'
    fps = 20
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()
