import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
import model_util, processing_util
import argparse

def input_arguments():
    parser = argparse.ArgumentParser(description='Input arguments for model creation and training.')
    parser.add_argument('data_dir', dest='data_dir', action='store', default='./flowers', help='Path to the parent file of data.')
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth', help='Path to save checkpoint.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Choose to use GPU (True/False)')
    parser.add_argument('--arch', dest='arch', action='store', default='resnet50', choices=['densenet121', 'resnet50', 'vgg16'], help='Choose an architecture for the model.')
    parser.add_argument('--hidden_units', dest='hidden_units', action='store', default=512, type=int, help='Enter the number of nodes in hidden layer.')
    parser.add_argument('--output_units', dest='output_units', action='store', default=102, type=int, help='Enter the number of nodes in output layer.')
    parser.add_argument('--dropout', dest='dropout', action='store', default=0.2, type=float, help='Enter the dropout probability.')
    parser.add_argument('--learn_rate', dest='learn_rate', action='store', default=0.003, type=float, help='Enter the learning rate of the model.')
    parser.add_argument('--epochs', dest='epochs', action='store', default=4, type=int, help='Enter the number of epochs.')
    parser.add_argument('--print_step', dest='print_step', action='store', default=10, type=int, help='Enter the number of training loops to take before validation.')
    parser.add_argument('to_test', dest='to_test', action='store_true', help='Choose to test the model on training dataset (True/False)')
    
    return parser.parse_args()

inpt_args = input_arguments()

data_dir = inpt_args.data_dir
save_dir = inpt_args.save_dir
gpu = inpt_args.gpu
arch = inpt_args.arch
hidden_units = inpt_args.hidden_units
output_units = inpt_args.output_units
dropout = inpt_args.dropout
learn_rate = inpt_args.learn_rate
epochs = inpt_args.epochs
print_step = inpt_args.print_step
to_test = inpt_args.to_test

device = processing_util.choose_gpu(gpu)
train_data, valid_data, test_data = model_util.create_datasets(data_dir)
trainloader, validloader, testloader = model_util.load_data(train_data, valid_data, test_data)
model, optimizer = model_util.model_setup(arch, hidden_units, output_units, dropout, learn_rate)
trained_model = model_util.train_model(device, model, trainloader, validloader, optimizer, epochs, print_step)
model_util.test_model(device, trained_model, testloader, to_test)
processing_util.save_checkpoint(trained_model, train_data, save_dir)