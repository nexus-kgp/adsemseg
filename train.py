import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from loaders.pascal_voc_loader import pascalVOCLoader
from loss import cross_entropy2d
from models.segmentor import fcn32s
from models.discriminator import LargeFOV 


def initialize_fcn32s(n_classes):

    segmentor = fcn32s

    try:
        segmentor = segmentor(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        segmentor.init_vgg16_params(vgg16)
    except:
        print('Error occured in initialising fcn32s')
        sys.exit(1)

    return segmentor


def train(epochs):

    # Setup Dataloader
    
    data_loader = pascalVOCLoader
    data_path = "/media/sangeet/Stuff/DC Shares/Datasets/VOCdevkit/VOC2012/"
    loader = data_loader(data_path, is_transform=True, img_size=(256, 256))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=1, num_workers=4, shuffle=True)

    # Setup Model for segmentor and discriminator
    segmentor = initialize_fcn32s(n_classes)

    discriminator = LargeFOV(n_class=21)


    # segmentor.cuda()

    # Setup optimizer for segmentor and discriminator
    optimizer = torch.optim.SGD(segmentor.parameters(), lr=1e-5, momentum=0.99, weight_decay=5e-4)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainloader):
            
            images = Variable(images)
            labels = Variable(labels)
            # images = Variable(images.cuda())
            # labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = segmentor(images)
            import pudb;pu.db
            discriminator_output = discriminator(outputs)

            loss = cross_entropy2d(outputs, labels)

            #TODO 
            # Now that the we have the forward propagation done its time to define\
            # the objective function to train

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, epochs, loss.data[0]))

        torch.save(segmentor, "{}.pkl".format(epoch))

if __name__ == '__main__':
    epochs = 100
    train(epochs)
