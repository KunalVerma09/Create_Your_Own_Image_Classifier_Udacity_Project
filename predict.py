import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np
import argparse
import processing_util

def input_arguments():
    parser = argparse.ArgumentParser(description='Input arguments for prediction.')
    parser.add_argument('img_path', dest='img_path', action='store', default='./flowers/test/1/image_06743.jpg', help='Path of the image.')
    parser.add_argument('checkpoint', dest='checkpoint', action='store', default='checkpoint.pth', help='Path to access saved checkpoint.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Choose to use GPU (True/False)')
    parser.add_argument('--top_k', dest='top_k', action='store', default=5, type=int, help='Choose number of top predictions.')
    parser.add_argument('--category_names', dest='cat_names', action='store', default='cat_to_name.json', help='Path to JSON file for category and name mapping.')
    parser.add_argument('display_class_name', dest='display_class_name', action='store_false', help='Choose to print class names (True/False)')
    
    return parser.parse_args()

inpt_args = input_arguments()

cat_names = inpt_args.cat_names
display_class_names = inpt_args.display_class_names
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

processing_util.display_class(cat_to_name, display_class_names)

img_path = inpt_args.img_path
checkpoint = inpt_args.checkpoint
gpu = inpt_args.gpu
top_k = inpt_args.top_k

device = processing_util.choose_gpu(gpu)
trained_model = processing_util.load_checkpoint(checkpoint)
processing_util.predict(img_path, trained_model, device, top_k, cat_to_name)