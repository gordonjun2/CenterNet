#!/usr/bin/env python
import os

import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets

from models.CenterNet_52 import model_52
from models.CenterNet_104 import model_104
from models.py_utils.kp import kp
from tensorboardX import SummaryWriter       # importing tensorboard
from torch.utils.data import DataLoader
from db.coco import MSCOCO
import cv2
import time
from utils.early_stopping import EarlyStopping
from test.coco_video import kp_detection
#import imageio
from test.coco_train import kp_detection_train

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CenterNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--es", dest="es", default=True, type=bool)

    #args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
    return args

def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def train(training_dbs, validation_db, validation_db_2, tb, suffix, cfg_file, es, start_iter=0):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)
    #validation_2_size = len(validation_db_2.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size)
    validation_queue = Queue(5)
    #validation_2_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)
    #pinned_validation_2_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data)
    sample_data = importlib.import_module(data_file).sample_data

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data, True)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)
        #validation_2_tasks = init_parallel_jobs([validation_db_2], validation_2_queue, sample_data, False)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    #validation_2_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()
    #validation_2_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    # validation_2_pin_args   = (validation_2_queue, pinned_validation_2_queue, validation_2_pin_semaphore)
    # validation_2_pin_thread = threading.Thread(target=pin_memory, args=validation_2_pin_args)
    # validation_2_pin_thread.daemon = True
    # validation_2_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(training_dbs[0])#, suffix)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    if es:
        early_stopping = EarlyStopping(patience=30, verbose=True)

    print("training start...")
    nnet.cuda()
    #nnet.cpu()

    #if suffix == 104:
    #    net = model_104(training_dbs[0])
    #    tb.add_graph(net, torch.rand(2, 3, 511, 511))#, torch.FloatTensor(training_dbs[0].db_inds))
    #elif suffix == 52:
    #    net = model_52(training_dbs[0])
    #    dummy_input = torch.randn(2, 3, 511, 511)
    #    tb.add_graph(net, dummy_input)
    #else:
    #    return
    #tb.close()

    ##### Model's Warm-up #####
    nnet.eval_mode()
    input = cv2.imread(training_dbs[0].image_file(0))
    start_time = time.time()
    detections = kp_detection(input, nnet, score_min=0.5) 
    end_time = time.time()
    infer_time = end_time - start_time
    print("\n##################################################")
    print("Warm-up + Inference Time: " + str(infer_time * 1000) + "ms")
    print("##################################################")
    ###########################

    ##### Model's Inference Time #####
    input = cv2.imread(training_dbs[0].image_file(0))
    start_time = time.time()
    detections = kp_detection(input, nnet, score_min=0.5) 
    end_time = time.time()
    infer_time = end_time - start_time
    print("\n##################################################")
    print("Inference Time: " + str(infer_time * 1000) + "ms")
    print("##################################################")
    ##################################

    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str("Training_Validation"), str("val2017"), str(suffix))      # Use MSCOCO 2017 for Validation in Training

    #if suffix is not None:
    #    result_dir = os.path.join(result_dir, suffix)

    make_dirs([result_dir])
    
    nnet.train_mode()

    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            #start_time = time.time()
            training_loss, focal_loss, pull_loss, push_loss, regr_loss = nnet.train(**training)
            #end_time = time.time()
            #infer_time = end_time - start_time
            #training_loss, focal_loss, pull_loss, push_loss, regr_loss, cls_loss = nnet.train(**training)

            #print("\nTotal Time per Iteration:" + str(infer_time) + "ms")
            #tb.add_scalar('Total Time (ms) vs Iteration', infer_time * 1000, iteration)

            if display and iteration % display == 0:
                print("\ntraining loss at iteration {}: {}".format(iteration, training_loss.item()))
                print("focal loss at iteration {}:    {}".format(iteration, focal_loss.item()))
                print("pull loss at iteration {}:     {}".format(iteration, pull_loss.item())) 
                print("push loss at iteration {}:     {}".format(iteration, push_loss.item()))
                print("regr loss at iteration {}:     {}".format(iteration, regr_loss.item()))
                #print("cls loss at iteration {}:      {}\n".format(iteration, cls_loss.item()))

            tb.add_scalar('Training Loss vs Iteration', training_loss.item(), iteration)
            tb.add_scalar('Focal Loss vs Iteration', focal_loss.item(), iteration)
            tb.add_scalar('Pull Loss vs Iteration', pull_loss.item(), iteration)
            tb.add_scalar('Push Loss vs Iteration', push_loss.item(), iteration)
            tb.add_scalar('Offset Loss vs Iteration', regr_loss.item(), iteration)
            #tb.add_scalar('Class Loss vs Iteration', cls_loss.item(), iteration)

            del training_loss, focal_loss, pull_loss, push_loss, regr_loss#, cls_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("\n##################################################")
                print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
                print("##################################################")
                tb.add_scalar('Validation Loss vs Iteration', validation_loss.item(), iteration)

                if es:
                    early_stopping(validation_loss, iteration, nnet, cfg_file)
                
                nnet.train_mode()

            epoch = len(training_dbs[0].db_inds) // system_configs.batch_size
            #print(epoch)

            if iteration % epoch == 0:     # Enter every epoch
                nnet.eval_mode()
                stats = kp_detection_train(validation_db_2, nnet, result_dir)
                map_avg = stats[0]
                map_50 = stats[1]
                map_75 = stats[2]
                map_small = stats[3]
                map_medium = stats[4]
                map_large = stats[5]
                mar_1 = stats[6]
                mar_10 = stats[7]
                mar_100 = stats[8]
                mar_small = stats[9]
                mar_medium = stats[10]
                mar_large = stats[11]
                tb.add_scalar('Average mAP vs Epoch', map_avg, epoch)
                tb.add_scalar('mAP (IoU 0.5) vs Epoch', map_50, epoch)
                tb.add_scalar('mAP (IoU 0.75) vs Epoch', map_75, epoch)
                tb.add_scalar('mAP (Area = Small) vs Epoch', map_small, epoch)
                tb.add_scalar('mAP (Area = Medium) vs Epoch', map_medium, epoch)
                tb.add_scalar('mAP (Area = Large) vs Epoch', map_large, epoch)
                tb.add_scalar('mAR (Max Detection = 1) vs Epoch', mar_1, epoch)
                tb.add_scalar('mAR (Max Detection = 10) vs Epoch', mar_10, epoch)
                tb.add_scalar('mAR (Max Detection = 100) vs Epoch', mar_100, epoch)
                tb.add_scalar('mAR (Area = Small) vs Epoch', mar_small, epoch)
                tb.add_scalar('mAR (Area = Medium) vs Epoch', mar_medium, epoch)
                tb.add_scalar('mAR (Area = Large) vs Epoch', mar_large, epoch)
                nnet.train_mode()

            if es and early_stopping.early_stop:
                print("Early stopping")
                break

            if not es:
                if iteration % snapshot == 0:          
                    nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    if args.cfg_file == "CenterNet-104":
        suffix = 104
    elif args.cfg_file == "CenterNet-52":
        suffix = 52
    else:
        print("~~~~~ Haha, you typed the model incorrectly! ~~~~~")
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    val_split_2   = system_configs.val_split_2

    print("loading all datasets...")
    dataset = system_configs.dataset
    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)
    validation_db_2 = datasets[dataset](configs["db"], val_split_2)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(training_dbs[0].configs)

    tb = SummaryWriter(comment=' Model = <Default>, batch_size = ' + str(system_configs.batch_size) +
                        ', learning_rate = ' + str(system_configs.learning_rate) +
                        ', decay_rate = ' + str(system_configs.decay_rate) + 
                        ', pull_weight = ' + str(system_configs.pull_weight) + 
                        ', push_weight = ' + str(system_configs.push_weight) + 
                        ', regr_weight = ' + str(system_configs.regr_weight))

    print("len of db: {}".format(len(training_dbs[0].db_inds)))
    train(training_dbs, validation_db, validation_db_2, tb, suffix, args.cfg_file, args.es, args.start_iter)

