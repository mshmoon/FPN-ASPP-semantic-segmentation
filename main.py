import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from argparse import ArgumentParser
import os
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage,Resize,RandomRotation,RandomGrayscale
import torchvision.datasets as datasets
import torch.nn.functional as F
from piwise.dataset import VOC12
from piwise.dataset import test_set
from piwise.cityscapes import CityScapes
from piwise.network import SegNet,PSPNet
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize

import pydensecrf.densecrf as dcrf
from piwise.metrics import runningScore
import torchvision.transforms as transforms
from PIL import Image

import sys
import time

NUM_CHANNELS = 3
NUM_CLASSES = 22

to_tensor=transforms.ToTensor()
color_transform = Colorize()
image_transform = ToPILImage()
to_img=transforms.ToPILImage()

 
input_transform = Compose([ 
    
    RandomGrayscale(0.02),
    CenterCrop((512,512)),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

target_transform = Compose([
    CenterCrop((512,512)), 
    ToLabel(),
    Relabel(255, 22),
    ])

)

    
def train(args, model):

    model.train()   
    weight = torch.ones(22)
    weight[0] = 0
   
        

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)  

    model.load_state_dict(torch.load(args.model_para),strict=True)
   
   
    total_step=0
    for epoch in range(0, args.num_epochs):

    
        loader = DataLoader(VOC12(args.datadir, input_transform, target_transform),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
 
        lr=args.learning_rate*((args.num_epochs-epoch)/args.num_epochs)

        epoch_loss = []
        optimizer=SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=args.weight_decay)


        for step, (images,labels,path) in enumerate(loader):           
      
            inputs = Variable(images).cuda()
            targets = Variable(labels).cuda()
            
            outputs= model(inputs) 
      
            def loss_back(loss,optimizer,args):  
                optimizer.zero_grad() 
                
                

                loss.backward()         
                optimizer.step()     
                epoch_loss.append(loss.data[0])              
                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss) / len(epoch_loss)
                    print(f'loss: {average} (epoch: {epoch}, step: {step})')  
                    
                if (epoch+1)%1 ==0:
                    if args.steps_save > 0 and step % args.steps_save == 0:
                       filename = f'{args.model}-{epoch:03}-{step:04}.pth'
                       torch.save(model.state_dict(), filename)                    
                       print(f'save: {filename} (epoch: {epoch}, step: {step})')
     
                
           
            loss = criterion(outputs,targets.squeeze())                                    
            loss_back(loss,optimizer,args)
            total_step +=1
            if args.steps_loss > 0 and step % args.steps_loss == 0:   
                print('-----------------------------------------')

    
def evaluate( model):

    model.load_state_dict(torch.load(args.model_para))
    loader = DataLoader(test_set('data/', input_transform, target_transform),
        num_workers=0, batch_size=b, shuffle=False)
    

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    for i ,(image,path) in enumerate(loader):
        image=Variable(image.cuda(), volatile=True)
        start =time.clock()   
        outputs=model(image)
        end =time.clock()
 


def main(args):
    Net = None
   
    if args.model == 'deeplab_fpn':
        Net = SegNet
    assert Net is not None, f'model {args.model} not available'

    model = Net(NUM_CLASSES)

    if args.cuda:
        model = model.cuda()
    if args.state:
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))

    if args.mode == 'eval':
        evaluate(model)

    if args.mode == 'train':
        train(args, model)
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
   
 
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--batch-size', type=int, default=1)
    parser_train.add_argument('--steps-loss', type=int, default=50)
    parser_train.add_argument('--learning_rate', type=float, default=0.001)
    parser_train.add_argument('--weight_decay', type=float, default=0.0005)
    parser_train.add_argument('--model_para', type=str, default=None)

    main(parser.parse_args())
