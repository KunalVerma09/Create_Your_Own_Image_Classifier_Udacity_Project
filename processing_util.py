import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np

def display_class(class_dict=cat_to_name, display=False):
    '''
    Arguments:
    - class_dict: dictionary with "class values"(key): "category name"(value) pairs, default=cat_to_name
    - display: choose to print class names, default=False
    
    return:
    None
    '''
    if display:
        print(cat_to_name)

def save_checkpoint(trained_model, train_data, filepath="checkpoint.pth"):
    '''
    Arguments:
    - trained_model: model trained on training dataset
    - train_data: training dataset
    - filepath: path of the file in which you wish to save the checkpoint, default="checkpoint.pth"
    
    return:
    None
    '''
    trained_model.class_to_idx = train_data.class_to_idx
    checkpoint = {"architecture": trained_model,
                  "model_state_dict": trained_model.state_dict(),
                  "class_to_idx": trained_model.class_to_idx}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath="checkpoint.pth"):
    '''
    Arguments:
    - filepath: path of the file in which you saved the checkpoint, default="checkpoint.pth"
    
    return:
    - trained_model: the saved model you trained on training dataset
    '''
    
    checkpoint = torch.load(filepath)
    trained_model = checkpoint["architecture"]
    trained_model.load_state_dict(checkpoint["model_state_dict"])
    trained_model.class_to_idx = checkpoint["class_to_idx"]
    
    return trained_model

def process_image(image_path):
    '''
    Arguments:
    - image_path: path of image to be scaled, cropped and normalized for a PyTorch model
    
    return:
    - np_image: NumPy array of image
    '''
    PIL_image = Image.open(image_path)
    image_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    processed_image = image_transform(PIL_image)
    np_image = np.array(processed_image)
    
    return np_image

def predict(image_path, trained_model, device, topk=5, class_dict=cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Arguments:
    - image_path: path of the image file
    - trained_model: trained model to be used for prediction
    - device: available device you wish to use to predict ("cuda"/"cpu")
    - topk: top k number of classes to be predicted, default=5
    - class_dict: dictionary with "class values"(key): "category name"(value) pairs, default=cat_to_name
    
    return:
    None
    '''
    image = torch.from_numpy(process_image(image_path)).unsqueeze(0)
    trained_model, image = trained_model.to(device), image.to(device)
    model.eval()
    
    output = trained_model(image)
    ps = torch.exp(output)
    
    top_ps, top_class_index = torch.topk(ps, 5)
    probs = top_ps.tolist()[0]
    top_class_index = top_class_index.tolist()[0]
    
    index = []
    for i in range(len(trained_model.class_to_idx.items())):
        index.append(list(trained_model.class_to_idx.items())[i][0])
    
    classes = [(index[i], cat_to_name[i]) for i in top_class_index]

    print(f"Top {topk} probabilities:{probs}")
    print(f"Top {topk} classes:{classes}")

def choose_gpu(choice=True):
    '''
    Arguments:
    - choice: if you wish to use gpu, default=True
    
    return:
    - device: returns the choice of device based on availability
    '''
    if choice:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU unavailable")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    return device
