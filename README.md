# Extra Instructions!

Follow the original instruction below if you are not using OpenCV in any way.

## OpenCV Installation Notes

If you are using OpenCV, ensure that the PyTorch version (0.4.1) is installed in this environment.

Steps to get OpenCV working:

**1.** Once your "CenterNet" environment is created as shown below, install OpenCV using:

  ```
	conda install -c anaconda opencv
  ```

  If error (inconsistent environment) is shown, ignore and wait until you can update your packages. Type "y" to continue.
  Note that this step will cause PyTorch to downgrade to 0.4.0.

**2.** Update PyTorch 0.4.0 to 0.4.1 using:

  ```
	conda install pytorch==0.4.1 -c pytorch
  ```

**3.** When using video_demo.py for video inference, install seaborn and imageio using:

  ```
	conda install -c anaconda seaborn
	conda install -c anaconda imageio
  ```

  If you face an OpenCV Error (cv2.imshow not implemented), install:
	pip install opencv-contrib-python

**4.** Done and enjoy CenterNet!

## Real-Time Video Objection Detection Using CenterNet

**1.** Activate CenterNet Environment

**2.** To do real-time inference on a video, use:

  ```
  python video_demo.py --model <Select your model> --testiter <Enter '480000' if using pretrained model> --file <./path/to/video.mp4> --score <Remove bboxes based on this score> <--save>
  ```

  Example:

  ```
  python video_demo.py --model CenterNet-104 --testiter 480000 --file road.mp4 --score 0.5 --save
  ```    

  Tips:

  -> Available models to use are: 'CenterNet-104' (More accurate but slower in inference) and 'CenterNet-52' (Less accurate but faster in inference)

  -> Use '480000' in --testiter if you are using the pretrained model. Enter another value if your pretrained model was trained under that no. of iterations

  -> Bboxes are kept or removed based on the score indicated in --score. The value should be 0 <= score >= 1. For example, if '--score 0.5' is used, then bboxes with confidence scores less than 0.5 will not be shown at the output.

  -> Use '--save' if you want to save each recorded frame into .jpg images. The images will be saved under CenterNet/Video_Frames/To_Convert/.

**3.** If you typed '--save' to save the recorded frames, you may also want to convert the frames into a video (no lag but it's not real-time). Do not rename the images. With the images already saved in the CenterNet/Video_Frames/To_Convert/ folder, enter CenterNet/Video_Frames and type the command:

  ```
  python frames_to_video.py
  ```

**4.** The converted video will be generated in the same directory.

**5.** Done and enjoy CenterNet!

# [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189)
by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Song Bai](http://songbai.site/), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed CenterNet is available here. For more technical details, please refer to our [arXiv paper](https://arxiv.org/abs/1904.08189).**

**We thank [Princeton Vision & Learning Lab](https://github.com/princeton-vl) for providing the original implementation of [CornerNet](https://github.com/princeton-vl/CornerNet).**

**CenterNet is an one-stage detector which gets trained from scratch. On the MS-COCO dataset, CenterNet achieves an AP of 47.0%, which surpasses all known one-stage detectors, and even gets very close to the top-performance two-stage detectors.**

## Abstract

  In object detection, keypoint-based approaches often suffer a large number of incorrect object bounding boxes, arguably due to the lack of an additional look into the cropped regions. This paper presents an efficient solution which explores the visual patterns within each cropped region with minimal costs. We build our framework upon a representative one-stage keypoint-based detector named CornerNet.
Our approach, named CenterNet, detects each object as a triplet, rather than a pair, of keypoints, which improves both precision and recall. Accordingly, we design two customized modules named cascade corner pooling and center pooling, which play the roles of enriching information collected by both top-left and bottom-right corners and providing more recognizable information at the central regions, respectively. On the MS-COCO dataset, CenterNet achieves an AP of 47.0%, which outperforms all existing one-stage detectors by a large margin. Meanwhile, with a faster inference speed, CenterNet demonstrates quite comparable performance to the top-ranked two-stage detectors.

## Introduction

CenterNet is a framework for object detection with deep convolutional neural networks. You can use the code to train and evaluate a network for object detection on the MS-COCO dataset.

* It achieves state-of-the-art performance (an AP of 47.0%) on one of the most challenging dataset: MS-COCO.

* Our code is written in Python, based on [CornerNet](https://github.com/princeton-vl/CornerNet).

*More detailed descriptions of our approach and code will be made available soon.*

**If you encounter any problems in using our code, please contact Kaiwen Duan: kaiwen.duan@vipl.ict.ac.cn.**

## Architecture

![Network_Structure](https://github.com/Duankaiwen/CenterNet/blob/master/Network_Structure.jpg)

## Comparison with other methods

![Tabl](https://github.com/Duankaiwen/CenterNet/blob/master/Table1.png)

![Tabl](https://github.com/Duankaiwen/CenterNet/blob/master/Table2.png)

![Tabl](https://github.com/Duankaiwen/CenterNet/blob/master/Table3.png)

  In terms of speed, we test the inference speed of both CornerNet and CenterNet on a NVIDIA Tesla P100 GPU. We obtain that the average inference time of CornerNet511-104 (means that the resolution of input images is 511X511 and the backbone is Hourglass-104) is 300ms per image and that of CenterNet511-104 is 340ms. Meanwhile, using the Hourglass-52 backbone can speed up the inference speed. Our CenterNet511-52 takes an average of 270ms to process per image, which is faster and more accurate than CornerNet511-104.

## Preparation
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CenterNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CenterNet
```

## Compiling Corner Pooling Layers
```
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS
```
cd <CenterNet dir>/external
make
```

## Installing MS COCO APIs
```
cd <CenterNet dir>/data/coco/PythonAPI
make
```

## Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CenterNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CenterNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation

To train CenterNet-104:
```
python train.py CenterNet-104
```
We provide the configuration file (`CenterNet-104.json`) and the model file (`CenterNet-104.py`) for CenterNet in this repo. 

We also provide a trained model for `CenterNet-104`, which is trained for 480k iterations using 8 Tesla V100 (32GB) GPUs. You can download it from [BaiduYun CenterNet-104](https://pan.baidu.com/s/1OQwMAPLcZkHWbTzD28cxow) (code: bfko) or [Google drive CenterNet-104](https://drive.google.com/open?id=1GVN-YrgExbPPcmzn_Lkr49f2IKjodg15) and put it under `<CenterNet dir>/cache/nnet` (You may need to create this directory by yourself if it does not exist). If you want to train you own CenterNet, please adjust the batch size in `CenterNet-104.json` to accommodate the number of GPUs that are available to you.

To use the trained model:
```
python test.py CenterNet-104 --testiter 480000 --split <split>
```

To train CenterNet-52:
```
python train.py CenterNet-52
```
We provide the configuration file (`CenterNet-52.json`) and the model file (`CenterNet-52.py`) for CenterNet in this repo. 

We also provide a trained model for `CenterNet-52`, which is trained for 480k iterations using 8 Tesla V100 (32GB) GPUs. You can download it from [BaiduYun CenterNet-52](https://pan.baidu.com/s/1xZHB7jq7Hmi0qKu46qnotw) (code: 680t) or [Google Drive CenterNet-52](https://drive.google.com/open?id=14vJYw4P9sxDoltjp5zDkOS3QjUa2zZIP) and put it under `<CenterNet dir>/cache/nnet` (You may need to create this directory by yourself if it does not exist). If you want to train you own CenterNet, please adjust the batch size in `CenterNet-52.json` to accommodate the number of GPUs that are available to you.

To use the trained model:
```
python test.py CenterNet-52 --testiter 480000 --split <split>
```

We also include a configuration file for multi-scale evaluation, which is `CenterNet-104-multi_scale.json` and `CenterNet-52-multi_scale.json` in this repo, respectively. 

To use the multi-scale configuration file:
```
python test.py CenterNet-52 --testiter <iter> --split <split> --suffix multi_scale
```
or
```
python test.py CenterNet-104 --testiter <iter> --split <split> --suffix multi_scale
```
