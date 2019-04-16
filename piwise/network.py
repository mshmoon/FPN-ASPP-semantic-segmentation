import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import visdom
from torch.utils import model_zoo
#from torchvision import models

from piwise import  model
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
from piwise import inception
from torch.utils.checkpoint import checkpoint


class SegNet(nn.Module):###################################################################

##############    resenet152     #########################
    def __init__(self, num_classes):
        super().__init__()

        # should be vgg16bn but at the moment we have no pretrained bn models
        self.resnet152 = model.resnet34(pretrained=True)  
         
        self.conv_module=nn.MouduleList([nn.Conv2d(512/i,output,3,1,1) for i in [1,2,4,8]])
        self.bn_module=nn.ModuleList([nn.BatchNorm2d(output) for i in range(10)])

        self.encode=nn.ModuleList([nn.Conv2d(128,128,1,1,0) for i in range(6)])
        self.encode_bn=nn.ModuleList([nn.Conv2d(128)] for i in range(6))

        self.conv_dilation=nn.ModuleList([nn.Conv2d(128,128,3,1,i,i)for i in [3,6,12] for j in range(4) )])
        self.conv_dilation_bn=nn.ModuleList([nn.BatchNorm2d(128) for i in range(12)])
        self.relu = nn.ReLU(inplace=True)
  
        self.cut_dim=nn.Sequential(
            nn.Conv2d(128*4,128*4,3,1,1),
            nn.BatchNorm2d(128*4),
            nn.ReLU(inplace=True),
             
        )    

        self.final=nn.Sequential(
            nn.Conv2d(128*4,2,3,1,1),     
        )

###################### #################
       
    def forward(self,x):
     
        x4,x3,x2,x1,x0=self.resnet152(x)

 
    
        a_1=self.enc4_1(x4[:,0:256,:,:])
        b_1=self.enc3_1(x3[:,0:128,:,:])
        c_1=self.enc2_1(x2[:,0:64,:,:])
        d_1=self.enc1_1(x1[:,0:32,:,:])
        
        a_2=self.enc4_2(x4[:,256:512,:,:])
        b_2=self.enc3_2(x3[:,128:256,:,:])
        c_2=self.enc2_2(x2[:,64:128,:,:])
        d_2=self.enc1_2(x1[:,32:64,:,:])
     
        a=torch.cat([a_1,a_2],1)
        b=torch.cat([b_1,b_2],1)
        c=torch.cat([c_1,c_2],1)
        d=torch.cat([d_1,d_2],1)
         

        b=(F.upsample_bilinear(a,scale_factor=2)+self.enc_1(b))
        c=(F.upsample_bilinear(b,scale_factor=2)+self.enc_2(c))      
        d=(F.upsample_bilinear(c,scale_factor=2)+self.enc_3(d))
        
        d=self.relu(self.res1(d)+d)
       
   

        d_0=self.relu(self.conv_dilation_0_0(d)+d)
        d_0=self.relu(self.conv_dilation_0_1(d_0)+d_0) 
        
        d_3=self.relu(self.conv_dilation_3_0(d)+d)  
        d_3=self.relu(self.conv_dilation_3_1(d_3)+d_3)

        d_6=self.relu(self.conv_dilation_6_0(d)+d)
        d_6=self.relu(self.conv_dilation_6_1(d_6)+d_6) 
     
        d_12=self.relu(self.conv_dilation_12_0(d)+d)
        d_12=self.relu(self.conv_dilation_12_1(d_12)+d_12)
                   
        d=torch.cat([d_0,d_3,d_6,d_12],1)

        d=self.cut_dim(d)

        d=self.final(d)

   
        a=F.upsample_bilinear(d,scale_factor=4)

        return a

