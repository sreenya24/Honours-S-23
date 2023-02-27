'''
Evaluation using GD Scribbles
for all the datasets , just to provide
upper bound for all the metrics .
'''

import time

import random
import os
import csv
import sys
import json
import copy
import cv2
import pathlib
import numpy as np
import math
import argparse
from vit_pytorch import ViT
from einops import rearrange
from empatches import EMPatches
import numpy


import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import Transformer


from collections import defaultdict, OrderedDict
import itertools


# Importing from files
from jain_glue_pipeline_v1 import *
from models.scribblenet import ScribbleNet
from v0_stage_two import *


# Global Variables
THRESHOLD = 0.45 ## binarization threshold after the model output
SPLITSIZE =  256  ## your image will be divided into patches of 256x256 pixels
setting = "base"  ## choose the desired model size [small, base or large], depending on the model you want to use
patch_size = 8 ## choose your desired patch size [8 or 16], depending on the model you want to use
image_size =  (SPLITSIZE,SPLITSIZE)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if setting == 'base':
    encoder_layers = 6
    encoder_heads = 8
    encoder_dim = 768

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Instantiate twice and load 2 different weight checkpoints
v1 = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = encoder_dim,
    depth = encoder_layers,
    heads = encoder_heads,
    mlp_dim = 2048
)

v2 = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = encoder_dim,
    depth = encoder_layers,
    heads = encoder_heads,
    mlp_dim = 2048
)

# Binary Output
binaryModel = ScribbleNet(
    encoder = v1,
    decoder_dim = encoder_dim,
    decoder_depth = encoder_layers,
    decoder_heads = encoder_heads
)

# Scribble Output
scribbleModel = ScribbleNet(
    encoder = v2,
    decoder_dim = encoder_dim,
    decoder_depth = encoder_layers,
    decoder_heads = encoder_heads
)

def argumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datacodes',type=str,default='I2')
    parser.add_argument('--binary_net_weights',type=str,default = '/home2/sreenya/TransformerSeamsInference/imp_weights/ind_sd_binary.pt')
    parser.add_argument('--scribble_net_weights',type=str,default = '/home2/sreenya/TransformerSeamsInference/imp_weights/I2_FIXED.pt')
    parser.add_argument('--parentdatadir',type=str,default='/home2/sreenya/')
    args = parser.parse_args()
    parser.add_argument('--expdir', type=str,default='/home2/sreenya/GND_SCR_V0')
    args = parser.parse_args()
    parser.add_argument('--jsonFile',type=str,default=None)
    parser.add_argument('--vis', type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.expdir,exist_ok=True)
    return args

if __name__ == "__main__":
    # Fetch args
    args = argumentParser()
    correct = 0
    errors=0

    evalDict={'LineAcc':0,'pixelAcc':0,'IoU':0 ,'Precision':0,'Recall':0,'Dice':0,'HD':0,'AvgHD':0,'HD95':0}

    # Sending model to the device..
    scribbleModel=scribbleModel.to(device)
    binaryModel=binaryModel.to(device)

    # Model Weight Loadings ...
    binaryModel.load_state_dict(torch.load(args.binary_net_weights, map_location=device),strict=True)
    scribbleModel.load_state_dict(torch.load(args.scribble_net_weights, map_location=device),strict=True)

    # Read json
    with open('/home2/sreenya/ICDARTest/{}_DATA/{}_TEST/{}_TEST.json'.format(args.datacodes,args.datacodes,args.datacodes),'r') as f :
        data = json.load(f)

    N = len(data)

    dims = []
    ress = []
    ress2 = []
    final = []

    # JSON
    for i,inst in enumerate(data):
        save = False
        try:
            path = inst['imgPath'].replace('./',args.parentdatadir).replace('.tif','.jpg')
            scribbles=inst['scribbles']

            img = cv2.imread(path)
            gds = inst['gdPolygons']
            imgDims = [img.shape[0],img.shape[1]]
            _,imgName = os.path.split(path)

            dims.append(imgDims)
            start = time.time()
            # Stage I Call
            bimg = inferenceTestImageModel(binaryModel,path,PDIM=256,DIM=256,OVERLAP=0.10,save=True)
            end = time.time()
            ress.append(end-start)

            netScribbles = copy.deepcopy(scribbles)
            if netScribbles is None or len(netScribbles)==1 :
                print('Scribble generation failed , skipping !')
                continue

            start = time.time()
            # Stage II Call
            predpolys = imageTask(img,bimg,netScribbles)
            end = time.time()
            ress2.append(end-start)

            if predpolys is None :
                print('Polygon generation failed , skipping !')
                continue

            final.append([dims[-1], ress[-1], ress2[-1]])


        except Exception as exp :
            print('Error ! : {}'.format(exp))
            errors+=1
            continue

    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerows(final)