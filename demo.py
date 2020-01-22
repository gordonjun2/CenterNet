import argparse

import cv2

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory
from utils.drawer import Drawer

def load_model(weight_path):

    # TODO: use NetworkFactory to load model and put on eval mode
    # print("building neural network...")
    # nnet = NetworkFactory(db)
    # print("loading parameters...")
    # nnet.load_params(test_iter)

    # test_file = "test.{}".format(db.data)
    # testing = importlib.import_module(test_file).testing

    # nnet.cuda()
    # nnet.eval_mode()

    pass


def show_video(video_file, nnet):

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    # sample = 1 # every <sample> sec take one frame
    # sample_num = sample * fps


    if not cap.isOpened():
        print("Error in opening video stream or file")

    frame_count = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            # if sample_num%frame_count != 0:
            #     continue

            # do what you want
            # TODO get center and corner (nnet)
            # TODO user drawer on frame

            cv2.imshow("Frame", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo for thurs")
    parser.add_argument("-f", "--file_dir", help="video file path")
    parser.add_argument("-w", "--weight_path", help="weight file path")
    args = parser.parse_args()
    print(args.file_dir)

    nnet = load_model(args.weight_path)

    show_video(args.file_dir, nnet)
