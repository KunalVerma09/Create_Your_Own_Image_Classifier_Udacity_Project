import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def create_datasets(location="./flowers"):
    '''
    Arguments:
    - location: Location of the main folder containing training, validation and testing data, default="./flowers"
    
    return:
    - train_data: training dataset
    - valid_data: validation dataset
    - test_data: testing dataset
    '''
    data_dir = location
    train_dir = location + "/train"
    valid_dir = location + "/valid"
    test_dir = location + "/test"
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)
    
    return train_data, valid_data, test_data

def load_data(train_data, valid_data, test_data):
    '''
    Arguments:
    - train_data: training dataset
    - valid_data: validation dataset
    - test_data: testing dataset
    
    return:
    - trainloader: training dataloader
    - validloader: validation dataloader
    - testloader: testing dataloader
    '''
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader

def model_setup(arch="resnet50", hidden_units=512, output_units=102, dropout=0.2, learn_rate=0.003):
    '''
    Arguments:
    - arch: parent architecture you wish to choose ("densenet121", "resnet50", "vgg16"), default="resnet50"
    - hidden_units: number of nodes in the hidden layer, default=512
    - output_units: number of nodes in the output layer, default=102
    - dropout: dropout probability (0-1), default=0.2
    - learn_rate: learning rate of the model, default=0.003
    
    return:
    - model: adapted model for your dataset
    - optimizer: optimizer chosen to optimize model parameters
    '''
    input_features = {"densenet121": 1024, "resnet50": 2048, "vgg16": 25088}
    
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
        
    for param in model.parameters():
    param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(input_features[arch], hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=dropout),
                               nn.Linear(hidden_units, output_units),
                               nn.LogSoftmax(dim=1))
    
    if arch == "resnet50":
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
    else:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    return model, optimizer

def train_model(device, model, trainloader, validloader, optimizer, epochs=4, print_step=10):
    '''
    Arguments:
    - device: available device you wish to use to train ("cuda"/"cpu")
    - model: adapted model for your dataset
    - trainloader: training dataloader
    - validloader: validation dataloader
    - optimizer: optimizer chosen to optimize model parameters
    - epochs: number of epochs to run the model
    - print_step: number of training loops after which you want to validate
    
    return:
    - trained_model: model trained on the training dataset
    '''
    model.to(device)
    criterion = nn.NLLLoss()
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_step == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    logps = model(images)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch: {epoch+1}/{epochs}.. ",
                      f"Training loss: {running_loss/print_step:.3f}.. ",
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. ",
                      f"Accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
    
    trained_model = model
    
    return trained_model

def test_model(device, trained_model, testloader, to_test=True):
    '''
    Arguments:
    - device: available device you wish to use to test ("cuda"/"cpu")
    - trained_model: trained model for your dataset
    - testloader: testing dataloader
    - to_test: choose to test the model, default=True
    
    return:
    None
    '''
    if to_test:
        criterion = nn.NLLLoss()
        trained_model.eval()
        test_loss = 0
        accuracy = 0

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            logps = trained_model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        print(f"Test loss: {test_loss/len(testloader):.3f}..",
              f"Accuracy: {accuracy/len(testloader):.3f}")
