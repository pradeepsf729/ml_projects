import torch

import torch
import torch.nn as nn

import sys

from PIL import Image
from torchvision import transforms

'''
Beow two classes are required to use ResNet to use for GAN's
Using it as it is retrieved from book for now.
'''
class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)

if len(sys.argv[1]) <= 1:
    print('Provide input image : ')
    print('Example : > python gane_test.py horse.jpeg')
    sys.exit(0)

input_file = sys.argv[1]

# initialize the network with horsetozebra generation trained weights
netG = ResNetGenerator()
model_data = torch.load("horse2zebra_0.4.0.pth") # pre-trained weight file is downloaded.
netG.load_state_dict(model_data)

# this will initliaze and do setup
netG.eval()

# process the input image and convert to tensor.
preprocess = transforms.Compose([transforms.Resize(256), # cropping to 256 * 256
                                transforms.ToTensor()])

img = Image.open(input_file)
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

print('Using input horse image of shape : ', batch_t.shape)

# perform the inference with the GAN network with the input image.
batch_out = netG(batch_t)

print('Got the output zebra image of shape : ', batch_out.shape)

# convert the output tensor back to image (which is the generated tensor)
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)

output_file = input_file.split('.')[0] + '_gan_out.jpeg'
out_img.save(output_file)

print('Output file saved to : ', output_file)

out_img.show()