import argparse
import time
import cv2

from config import system_configs
from utils.drawer import Drawer                         # Import Drawer to add bboxes

import os
import torch
import pprint
import json
import importlib
import numpy as np
import matplotlib
from test.coco_video import kp_detection
from nnet.py_factory_video import NetworkFactory        # Import CenterNet Model
from db.detection_video import db_configs               # Import 'db' parameters


def show_video(video_file, nnet, drawer, score_min, save = False):                                # , debug): <--- UNTESTED (Another way of adding bboxes)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:" + str(fps))

    #sample = 0.5 # every <sample> sec take one frame                               # Use only if you do not want the infer every frame
    #sample_num = sample * fps

    if not cap.isOpened():
        print("Error in opening video stream or file")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            start_time = time.time()
            detections = kp_detection(frame, nnet, score_min)                       # , debug) <--- UNTESTED (Another way of adding bboxes)
            end_time = time.time()
            infer_time = end_time - start_time
            print("Inference Time:" + str(infer_time) + "s")
            # print("~~~~~Detections~~~~~")
            # print(detections)

            #if sample_num%frame_count != 0:                                        # Use only if you do not want the infer every frame
            #     continue

            # do what you want
            # TODO get center and corner (nnet)
            # TODO user drawer on frame
            
            frame_det = drawer.draw_dets_video(frame, detections, infer_time)
            cv2.imshow("Frame", frame_det)

            if save:
                cv2.imwrite('./Video_Frames/To_Convert/' + str(frame_count) + ".jpg", frame_det)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Demo")
    parser.add_argument("--model", dest="json_file", help="which .json file in ./confg", type=str)      # CenterNet-52 or CenterNet-104
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)                                                         # Used to identify pretrained model
    parser.add_argument("--file", dest="file_dir", help="video file path", type=str)                    # Path to video for detection
    parser.add_argument("--score", dest="score_min", help="Remove bboxes of those scores < score", 
                        type=float)                                                                     # Minimise bboxes
    parser.add_argument("--save", action="store_true")
    #parser.add_argument("--debug", action="store_true")                                                 
    args = parser.parse_args()

    print("Video File:" + str(args.file_dir))

    json_file = os.path.join(system_configs.config_dir, args.json_file + ".json")

    print("json_file: {}".format(json_file))

    with open(json_file, "r") as f:
        configs = json.load(f)                                                  # Read .json file to retrieve 'system' and 'db' parameters

    configs["system"]["snapshot_name"] = args.json_file                         # Insert model's name into configuration file
    system_configs.update_config(configs["system"])                             # Update config.py based on retrieved 'system' parameters
    db_configs.update_config(configs["db"])                                     # Update db/base.py based on retrieved 'db' parameters

    print("system config...")
    pprint.pprint(system_configs.full)                                          # Show 'system' parameters in terminal

    print("db config...")
    pprint.pprint(db_configs.full)                                              # Show 'db' parameters in terminal

    print("loading parameters at iteration: {}".format(args.testiter))          # Show args.testiter in terminal

    print("building neural network...")
    nnet = NetworkFactory()                                                     # Initialise CenterNet's neural network
    print("loading parameters...")
    nnet.load_params(args.testiter)                                             # To locate CenterNet's pretrained model

    drawer = Drawer()                                                           # Initialise Drawer to add bboxes in frames later

    #nnet.cpu()                                                                 # Uncomment if using cpu
    nnet.cuda()                                                                 # Comment if using cpu
    nnet.eval_mode()

    show_video(args.file_dir, nnet, drawer, args.score_min, args.save)                     # , args.debug) <--- UNTESTED (Another way of adding bboxes)
