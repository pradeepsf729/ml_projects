'''
Images recognition using Pytorch.
'''
import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d
from torchvision import datasets
from torchvision import transforms
from  torch.utils.data import DataLoader
import datetime
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        # first convolution and pooling layer
        conv_output = self.conv1(x)
        conv_act_output = torch.tanh(conv_output)
        out_1 = max_pool2d(conv_act_output, 2)

        # second convolution and pooling layer
        conv_output = self.conv2(out_1)
        conv_act_output = torch.tanh(conv_output)
        out_2 = max_pool2d(conv_act_output, 2)

        # flatten the maxpool layer output to give as input to Fully Connected layer.
        out = out_2.view(-1, 8 * 8 * 8)
        
        # first fully connected layer and activation layer.
        out = self.fc1(out)
        out = torch.tanh(out)

        # second fully connected layer and ouput layer.
        # -> outputs two values per image (two probabilities)
        out = self.fc2(out)

        return out

# data loading
IMAGE_DIR = 'images_data'
def load_data_and_transform(is_train=True):
    # load input data to calculate mean, std
    tensor_cifar10 = datasets.CIFAR10(
                        IMAGE_DIR,
                        train=is_train,
                        download=True,
                        transform=transforms.ToTensor())

    total_images = [img_t for img_t, _ in tensor_cifar10]
    imgs = torch.stack(total_images, dim=3)

    mean_tr = imgs.view(3, -1).mean(dim=1)
    std_tr = imgs.view(3, -1).std(dim=1)

    #loading input data and normalizing.
    transformed_cifar10 = datasets.CIFAR10(
                            IMAGE_DIR, 
                            train=is_train, 
                            download=True, 
                            transform=transforms.Compose(
                                                    [transforms.ToTensor(),
                                                    transforms.Normalize(mean_tr, std_tr)]))
    return transformed_cifar10

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    for epoch in range(1, n_epochs):
        loss_train = 0.0
        for imgs, labels in train_loader:
            # move data to GPU to make things faster
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

            loss_train += loss.item()
        
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))

def validate(model, traindata_loader, validatedata_loader):
    for (name, loader) in zip(['train', 'validate'], [traindata_loader, validatedata_loader]):
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:

                # move data to GPU
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)

                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))

transformed_cifar10 = load_data_and_transform(True)

# we need only labels for bird, airplane (labels - 0, 2)
# filter out other labels.

label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[img_label]) for img, img_label in transformed_cifar10 if img_label in [0, 2]]

# get the training data batch loader
# shuffled 64 images per batch
# it outputs a generator
traindata_loader = DataLoader(cifar2, batch_size=64, shuffle=True)

# train on GPU if available else on CPU
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# move model parameters to GPU if available
model = Net().to(device=device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

print(f"Training on device - {device}.")

training_loop(50, optimizer, model, loss_fn, traindata_loader, device)

# load the validation data
# do the same tranformation done for training data, normalization etc..
cifar2_val = load_data_and_transform(False)

# filter only 0, 2
label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2_val = [(img, label_map[img_label]) for img, img_label in cifar2_val if img_label in [0, 2]]

# predict the accuracy
val_loader = DataLoader(cifar2_val, batch_size=64, shuffle=True)

validate(model, traindata_loader, val_loader)

# saving model params to file directly

torch.save(model.state_dict(), 'birds_vs_airplanes.pt')

# loading the saving params

loaded_model = Net().to(device=device)
params_dict = torch.load('birds_vs_airplanes.pt', map_location=device)
loaded_model.load_state_dict(params_dict)
validate(loaded_model, traindata_loader, val_loader)
