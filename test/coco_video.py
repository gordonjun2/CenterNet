import os
import cv2
import pdb
import json
import copy
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

import sys
sys.path.append("../../")                                       # Adds higher directory to python modules path.
from db.detection_video import db_configs                       # Import 'db' parameters
from db.coco_video import mscoco_classes                        # Import 'class_name' function

colours = np.random.rand(80,3)

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    tx_inds = xs[:,:,0] <= -5
    bx_inds = xs[:,:,1] >= sizes[0,1]+5
    ty_inds = ys[:,:,0] <= -5
    by_inds = ys[:,:,1] >= sizes[0,0]+5
    
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
    detections[:,tx_inds[0,:],4] = -1
    detections[:,bx_inds[0,:],4] = -1
    detections[:,ty_inds[0,:],4] = -1
    detections[:,by_inds[0,:],4] = -1

def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

def kp_decode(nnet, images, K, ae_threshold=0.5, kernel=3):
    detections, center = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)

    detections = detections.data.cpu().numpy()
    center = center.data.cpu().numpy()

    return detections, center

def kp_detection(frame, nnet, score_min, debug=False, decode_func=kp_decode):
    #debug_dir = os.path.join(result_dir, "debug")
    #if not os.path.exists(debug_dir):
    #    os.makedirs(debug_dir)

    #print("Sanity check per frame")
    #print(frame)
    
    #if db.split != "trainval":
    #    db_inds = db.db_inds if debug else db.db_inds
    #else:
    #    db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]
    #num_images = db_inds.size
    #print(num_images)

    K             = db_configs.top_k
    ae_threshold  = db_configs.ae_threshold
    nms_kernel    = db_configs.nms_kernel
    
    scales        = db_configs.test_scales
    weight_exp    = db_configs.weight_exp
    merge_bbox    = db_configs.merge_bbox
    categories    = db_configs.categories
    nms_threshold = db_configs.nms_threshold
    max_per_image = db_configs.max_per_image
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db_configs.nms_algorithm]

    top_bboxes = {}
    #for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
    #db_ind = db_inds[ind]
    #print(db_ind)
    #image_id   = db.image_ids(db_ind)
    #print(image_id)
    #image_file = db.image_file(db_ind)
    #image_file = os.path.join(system_configs.data_dir, "coco", "images", "test", "{}").format("image" + str(db_ind + 1) + ".jpg")
    #image_id   = "image" + str(db_ind + 1) + ".jpg"
    #if db_ind < 9:
    #	image_id   = "00000000000" + str(db_ind + 1) + ".jpg"
    #	image_file = os.path.join(system_configs.data_dir, "coco", "images", "val2017", "{}").format("" + str(db_ind + 1) + ".jpg")
    #elif db_ind >= 9 and db_ind < 99:
    #	image_id   = "0000000000" + str(db_ind + 1) + ".jpg"
    #	image_file = os.path.join(system_configs.data_dir, "coco", "images", "val2017", "{}").format("0000000000" + str(db_ind + 1) + ".jpg")
    
    #print(image_id)
    #print(image_file)
    #image = cv2.imread(frame)
    image = frame
    height, width = image.shape[0:2]

    detections = []
    center_points = []

    for scale in scales:
        new_height = int(height * scale)
        new_width  = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width  = new_width  | 127

        images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios  = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes   = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio  = out_width  / inp_width

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)     # From CenterNet/db/coco.py
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)      # From CenterNet/db/coco.py

        resized_image = resized_image / 255.
        normalize_(resized_image, mean, std)

        images[0]  = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0]   = [int(height * scale), int(width * scale)]
        ratios[0]  = [height_ratio, width_ratio]       

        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        dets, center = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
        dets   = dets.reshape(2, -1, 8)
        center = center.reshape(2, -1, 4)
        dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
        center[1, :, [0]] = out_width - center[1, :, [0]]
        dets   = dets.reshape(1, -1, 8)
        center   = center.reshape(1, -1, 4)
        
        _rescale_dets(dets, ratios, borders, sizes)
        center [...,[0]] /= ratios[:, 1][:, None, None]
        center [...,[1]] /= ratios[:, 0][:, None, None] 
        center [...,[0]] -= borders[:, 2][:, None, None]
        center [...,[1]] -= borders[:, 0][:, None, None]
        np.clip(center [...,[0]], 0, sizes[:, 1][:, None, None], out=center [...,[0]])
        np.clip(center [...,[1]], 0, sizes[:, 0][:, None, None], out=center [...,[1]])
        dets[:, :, 0:4] /= scale
        center[:, :, 0:2] /= scale

        if scale == 1:
            center_points.append(center)
        detections.append(dets)

    detections = np.concatenate(detections, axis=1)
    center_points = np.concatenate(center_points, axis=1)

    classes    = detections[..., -1]
    classes    = classes[0]
    detections = detections[0]
    center_points = center_points[0]
    
    valid_ind = detections[:,4]> -1
    valid_detections = detections[valid_ind]
    
    box_width = valid_detections[:,2] - valid_detections[:,0]
    box_height = valid_detections[:,3] - valid_detections[:,1]
    
    s_ind = (box_width*box_height <= 22500)
    l_ind = (box_width*box_height > 22500)
    
    s_detections = valid_detections[s_ind]
    l_detections = valid_detections[l_ind]
    
    s_left_x = (2*s_detections[:,0] + s_detections[:,2])/3
    s_right_x = (s_detections[:,0] + 2*s_detections[:,2])/3
    s_top_y = (2*s_detections[:,1] + s_detections[:,3])/3
    s_bottom_y = (s_detections[:,1]+2*s_detections[:,3])/3
    
    s_temp_score = copy.copy(s_detections[:,4])
    s_detections[:,4] = -1
    
    center_x = center_points[:,0][:, np.newaxis]
    center_y = center_points[:,1][:, np.newaxis]
    s_left_x = s_left_x[np.newaxis, :]
    s_right_x = s_right_x[np.newaxis, :]
    s_top_y = s_top_y[np.newaxis, :]
    s_bottom_y = s_bottom_y[np.newaxis, :]
    
    ind_lx = (center_x - s_left_x) > 0
    ind_rx = (center_x - s_right_x) < 0
    ind_ty = (center_y - s_top_y) > 0
    ind_by = (center_y - s_bottom_y) < 0
    ind_cls = (center_points[:,2][:, np.newaxis] - s_detections[:,-1][np.newaxis, :]) == 0
    ind_s_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
    index_s_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_s_new_score], axis = 0)
    s_detections[:,4][ind_s_new_score] = (s_temp_score[ind_s_new_score]*2 + center_points[index_s_new_score,3])/3
    
    l_left_x = (3*l_detections[:,0] + 2*l_detections[:,2])/5
    l_right_x = (2*l_detections[:,0] + 3*l_detections[:,2])/5
    l_top_y = (3*l_detections[:,1] + 2*l_detections[:,3])/5
    l_bottom_y = (2*l_detections[:,1]+3*l_detections[:,3])/5
    
    l_temp_score = copy.copy(l_detections[:,4])
    l_detections[:,4] = -1
    
    center_x = center_points[:,0][:, np.newaxis]
    center_y = center_points[:,1][:, np.newaxis]
    l_left_x = l_left_x[np.newaxis, :]
    l_right_x = l_right_x[np.newaxis, :]
    l_top_y = l_top_y[np.newaxis, :]
    l_bottom_y = l_bottom_y[np.newaxis, :]
    
    ind_lx = (center_x - l_left_x) > 0
    ind_rx = (center_x - l_right_x) < 0
    ind_ty = (center_y - l_top_y) > 0
    ind_by = (center_y - l_bottom_y) < 0
    ind_cls = (center_points[:,2][:, np.newaxis] - l_detections[:,-1][np.newaxis, :]) == 0
    ind_l_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
    index_l_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_l_new_score], axis = 0)
    l_detections[:,4][ind_l_new_score] = (l_temp_score[ind_l_new_score]*2 + center_points[index_l_new_score,3])/3
    
    detections = np.concatenate([l_detections,s_detections],axis = 0)
    detections = detections[np.argsort(-detections[:,4])] 
    classes   = detections[..., -1]
            
    #for i in range(detections.shape[0]):
    #   box_width = detections[i,2]-detections[i,0]
    #   box_height = detections[i,3]-detections[i,1]
    #   if box_width*box_height<=22500 and detections[i,4]!=-1:
    #     left_x = (2*detections[i,0]+1*detections[i,2])/3
    #     right_x = (1*detections[i,0]+2*detections[i,2])/3
    #     top_y = (2*detections[i,1]+1*detections[i,3])/3
    #     bottom_y = (1*detections[i,1]+2*detections[i,3])/3
    #     temp_score = copy.copy(detections[i,4])
    #     detections[i,4] = -1
    #     for j in range(center_points.shape[0]):
    #        if (classes[i] == center_points[j,2])and \
    #           (center_points[j,0]>left_x and center_points[j,0]< right_x) and \
    #           ((center_points[j,1]>top_y and center_points[j,1]< bottom_y)):
    #           detections[i,4] = (temp_score*2 + center_points[j,3])/3
    #           break
    #   elif box_width*box_height > 22500 and detections[i,4]!=-1:
    #     left_x = (3*detections[i,0]+2*detections[i,2])/5
    #     right_x = (2*detections[i,0]+3*detections[i,2])/5
    #     top_y = (3*detections[i,1]+2*detections[i,3])/5
    #     bottom_y = (2*detections[i,1]+3*detections[i,3])/5
    #     temp_score = copy.copy(detections[i,4])
    #     detections[i,4] = -1
    #     for j in range(center_points.shape[0]):
    #        if (classes[i] == center_points[j,2])and \
    #           (center_points[j,0]>left_x and center_points[j,0]< right_x) and \
    #           ((center_points[j,1]>top_y and center_points[j,1]< bottom_y)):
    #           detections[i,4] = (temp_score*2 + center_points[j,3])/3
    #           break
    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]

    #top_bboxes[image_id] = {}
    top_bboxes = {}

    for j in range(categories):
        keep_inds = (classes == j)

        #top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)

        if merge_bbox:

            #soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
            soft_nms_merge(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)

        else:

            #soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
            soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)

        #top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

    scores = np.hstack([

        #top_bboxes[image_id][j][:, -1] 
        top_bboxes[j][:, -1]

        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):

            #keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
            keep_inds = (top_bboxes[j][:, -1] >= thresh)

            #top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]
            top_bboxes[j] = top_bboxes[j][keep_inds]

    if debug:
        #image_file = db.image_file(db_ind)
        #image      = cv2.imread(image_file)

        #image      = cv2.imread(frame)

        im         = image[:, :, (2, 1, 0)]
        fig, ax    = plt.subplots(figsize=(12, 12)) 
        fig        = ax.imshow(im, aspect='equal')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        #bboxes = {}
        for j in range(1, categories + 1):

            #keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.4)
            keep_inds = (top_bboxes[j][:, -1] >= 0.4)

            #cat_name  = db.class_name(j)
            cat_name  = mscoco_classes.class_name(j)

            n = -1

            #for bbox in top_bboxes[image_id][j][keep_inds]:
            for bbox in top_bboxes[j][keep_inds]:

                n = n + 1  
                #print(top_bboxes[image_id][j][keep_inds][n][4])  
                bbox  = bbox[0:4].astype(np.int32)
                xmin     = bbox[0]
                ymin     = bbox[1]
                xmax     = bbox[2]
                ymax     = bbox[3]
                #if (xmax - xmin) * (ymax - ymin) > 5184:
                ax.add_patch(plt.Rectangle((xmin, ymin),xmax - xmin, ymax - ymin, fill=False, edgecolor= colours[j-1], 
                            linewidth=4.0))

                #ax.text(xmin+1, ymin-3, '{:s}: {:f}'.format(cat_name, top_bboxes[image_id][j][keep_inds][n][4]), bbox=dict(facecolor= colours[j-1], ec='black', lw=2,alpha=0.5),
                #        fontsize=15, color='white', weight='bold')
                ax.text(xmin+1, ymin-3, '{:s}: {:f}'.format(cat_name, top_bboxes[j][keep_inds][n][4]), bbox=dict(facecolor= colours[j-1], ec='black', lw=2,alpha=0.5), fontsize=15, color='white', weight='bold')


        #debug_file1 = os.path.join(debug_dir, "{}.pdf".format(db_ind))
        #debug_file2 = os.path.join(debug_dir, "{}.jpg".format(db_ind))
        #plt.savefig(debug_file1)
        #plt.savefig(debug_file2)
        #plt.close()
        #cv2.imwrite(debug_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    #result_json = os.path.join(result_dir, "results.json")
    #detections  = db.convert_to_coco(top_bboxes)
    #with open(result_json, "w") as f:
    #    json.dump(detections, f)

    #cls_ids   = list(range(1, categories + 1))
    #image_ids = [db.image_ids(ind) for ind in db_inds]
    #db.evaluate(result_json, cls_ids, image_ids)
    #print(top_bboxes)
    detections = mscoco_classes.convert_to_coco(top_bboxes, score_min)

    return detections

def testing(frame, nnet, score_min, debug=False):
    return globals()[system_configs.sampling_function](frame, nnet, score_min, debug=debug)
