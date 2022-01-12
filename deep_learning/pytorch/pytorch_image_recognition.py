import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from  torch.utils.data import DataLoader

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

transformed_cifar10 = load_data_and_transform(True)

# we need only labels for bird, airplane (labels - 0, 2)
# filter out other labels.

label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[img_label]) for img, img_label in transformed_cifar10 if img_label in [0, 2]]
#cifar2_val = [(img, label_map[img_label]) for img, img_label in cifar10_val if img_label in [0, 2]]

model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)

model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2)
)

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

n_epochs = 100

# get the training data batch loader
traindata_loader = DataLoader(cifar2, batch_size=64, shuffle=True)

softmax = nn.Softmax(dim=1)
for epoch in range(1, n_epochs):
    for imgs, labels in traindata_loader:
        batch_size = imgs.shape[0]
        out = model(imgs.view(batch_size, -1))
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


# load the validation data
# do the same tranformation done for training data, normalization etc..
cifar2_val = load_data_and_transform(False)

# filter only 0, 2
label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2_val = [(img, label_map[img_label]) for img, img_label in cifar2_val if img_label in [0, 2]]


# predict the accuracy
val_loader = DataLoader(cifar2_val, batch_size=64, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        softmax(outputs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

    print("Accuracy: %f", correct / total)

print('Total parameters in the model : ', end=' ')

total_params = [p.numel() for p in model.parameters() if p.requires_grad==True]

print(sum(total_params))




