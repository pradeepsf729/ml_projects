'''
uses resnet101 network with various images
'''

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import sys

# load the resnet model
# pre-trained indicates the weights and biases are loaded into the object.
resnet = models.resnet101(pretrained=True)

# take the input image provided.
if len(sys.argv) <= 1:
    print('Run the command with an input image file \n')
    print('Example : > python test_resnet101.py dog.jpg')
    sys.exit(0)

# load the input image
image_file = sys.argv[1]
img = Image.open(image_file)

# we need to transform any input images to the format
# that resnet netowrk is compatible with (or trained with)
preprocess = transforms.Compose([
                transforms.Resize(256),           # crop the image to 256 * 256
                transforms.CenterCrop(224),       # crop corners further to 224 * 224
                transforms.ToTensor(),            # convert to pytorch tensor.
                transforms.Normalize(             # normalize the RGB values
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

print('Processing image of shape ', img_t.shape)

# for some pre-trained models, some initialization 
# is required to load some layers.
resnet.eval()

# run the forward pass of the network with the image
output = resnet(batch_t)

# the output will contain prediction value for each of the classes (1000 in this case)
print('Output shape after forward pass ', output.shape, '\n')

# we will perform softmax to get the probabilities (confidence score for each class)
# given by the network.
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100

# sorting the output
_, indices = torch.sort(output, descending=True)

# load the output labels/classes from the imagenet database.
# this is the same data that resnet is trained on.
with open('imagenet_classes.txt') as f:
    training_labels = [line.strip() for line in f.readlines()]

print('Prediction :', end=' ')
idx = indices[0][0]
print(training_labels[idx])
print('Confidence : ', percentage[idx].item())

print('-----------------------------\n')
print('top 5 predictions\n')
print('Image label', ' -> Confidence percentage')
result = {}
for idx in indices[0][:5]:
    print(training_labels[idx], ' -> ', percentage[idx].item())

