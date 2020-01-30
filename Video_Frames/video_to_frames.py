import cv2
import os

vidcap = cv2.VideoCapture('road_2.mp4')

if not os.path.exists('./To_Convert'):
    os.makedirs('./To_Convert')

print(os.getcwd())

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()

    if hasFrames:
        os.chdir('./To_Convert')
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
        os.chdir('../')
    return hasFrames

sec = 0
frameRate = 0.01 #//it will capture image in each 0.01 second
count=1
success = getFrame(sec)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)