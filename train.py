import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

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
        # segmentor.init_vgg16_params(vgg16)
    except:
        print('Error occured in initialising fcn32s')
        sys.exit(1)

    return segmentor

batch_size = 1
use_gpu = torch.cuda.is_available()

segmentor = initialize_fcn32s(21)
discriminator = LargeFOV(n_class=21)

if use_gpu:
    zeros = Variable(torch.zeros((batch_size)).cuda(), requires_grad=False)
    ones = Variable(torch.ones((batch_size)).cuda(), requires_grad=False)
    segmentor.cuda()
    discriminator.cuda()
else:
    zeros = Variable(torch.zeros((batch_size)), requires_grad=False)
    ones = Variable(torch.ones((batch_size)), requires_grad=False)

d_loss = nn.BCELoss(size_average=False)

# Setup Model for segmentor and discriminator


g_optim = optim.Adam(segmentor.parameters(), lr=1e-5)

d_optim = optim.Adam(discriminator.parameters(), lr=1e-5)

# g = None

fake_loss_d = []
real_loss_d = []
real_loss_gen = []


def train(epochs):

    # Setup Dataloader
    
    data_loader = pascalVOCLoader
    data_path = "/media/sangeet/Stuff/DC Shares/Datasets/VOCdevkit/VOC2012/"
    loader = data_loader(data_path, is_transform=True, img_size=(256, 256))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=1, num_workers=4, shuffle=True)




    # segmentor.cuda()

    # Setup optimizer for segmentor and discriminator
    # optimizer = torch.optim.SGD(segmentor.parameters(), lr=1e-5, momentum=0.99, weight_decay=5e-4)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainloader):

            if use_gpu:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            import pudb;pu.db

            fake_out = segmentor(images)

            discriminator.zero_grad()
            segmentor.zero_grad()
            
            d_fake_out = discriminator(fake_out)
            fake_err = d_loss(d_fake_out, zeros)
            fake_err.backward(retain_graph=True)
            fake_loss_d.append(fake_err[0].clone().cpu().data.numpy()[0])


            d_real_out = discriminator(labels.float())
            real_err = d_loss(d_real_out, ones)
            real_err.backward()
            real_loss_d.append(real_err[0].clone().cpu().data.numpy()[0])
            d_optim.step()


            

            g_err = cross_entropy2d(fake_out, labels) + 0.65*(d_loss(d_fake_out,ones))
            g_err.backward()
            real_loss_gen.append(g_err[0].clone().cpu().data.numpy()[0])
            g_optim.step()




            #TODO 
            # Now that the we have the forward propagation done its time to define\
            # the objective function to train

            # if (i+1) % 20 == 0:
            #     print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, epochs, loss.data[0]))

        # torch.save(segmentor, "{}.pkl".format(epoch))

if __name__ == '__main__':
    epochs = 100
    train(epochs)
