import torch
import torch.nn as nn
import torch.nn.functional as F


class unetConv2(nn.Module):
    def __init__(self,in_channels,out_channels,is_batchnorm):
        super(unetConv2,self).__init__()

        if is_batchnorm:
            self.conv1=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.conv2=nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv2=nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True)
            )
    def forward(self, inputs):
        outputs=self.conv1(inputs)
        outputs=self.conv2(outputs)

        return outputs

class unetUp(nn.Module):
    def __init__(self,in_channels,out_channels,is_deconv):
        super(unetUp,self).__init__()
        self.conv=unetConv2(in_channels,out_channels,True)
        if is_deconv:
            self.up=nn.Sequential(
                nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.up=nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs1,inputs2):
        outputs2=self.up(inputs2)
        offset=outputs2.size()[2]-inputs1.size()[2]
        padding=2*[offset//2,offset//2]
        outputs1=F.pad(inputs1,padding) 

        return self.conv(torch.cat([outputs1,outputs2],1))

class unet(nn.Module):
    def __init__(self,feature_scale=2,in_channels=1,is_deconv=True,is_batchnorm=True):
        super(unet,self).__init__()
        self.is_deconv=is_deconv
        self.in_channels=in_channels
        self.is_batchnorm=is_batchnorm
        self.feature_scale=feature_scale

        filters=[64,128,256,512,1024]
        filters=[int(x/self.feature_scale) for x in filters]
        self.conv1=unetConv2(self.in_channels,filters[0],self.is_batchnorm)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.conv2=unetConv2(filters[0],filters[1],self.is_batchnorm)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        self.conv3=unetConv2(filters[1],filters[2],self.is_batchnorm)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)

        self.conv4=unetConv2(filters[2],filters[3],self.is_batchnorm)
        self.maxpool4=nn.MaxPool2d(kernel_size=2)

        self.center=unetConv2(filters[3],filters[4],self.is_batchnorm)
        self.up_concat4=unetUp(filters[4],filters[3],self.is_deconv)
        self.up_concat3=unetUp(filters[3],filters[2],self.is_deconv)
        self.up_concat2=unetUp(filters[2],filters[1],self.is_deconv)
        self.up_concat1=unetUp(filters[1],filters[0],self.is_deconv)

        self.final_1 = nn.Sequential(
                  nn.Conv2d(filters[0],1,kernel_size=3,stride=1,padding=1),
                  nn.BatchNorm2d(1),
                  nn.ReLU(inplace=True),
        )
        self.final_2 = nn.Sequential(
                  nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0),
                  nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        conv1=self.conv1(inputs)
        maxpool1=self.maxpool1(conv1)
        
        conv2=self.conv2(maxpool1)
        maxpool2=self.maxpool2(conv2)
        
        conv3=self.conv3(maxpool2)
        maxpool3=self.maxpool3(conv3)
        
        conv4=self.conv4(maxpool3)
        maxpool4=self.maxpool4(conv4)
        
        center=self.center(maxpool4)
        
        up4=self.up_concat4(conv4,center)
        
        up3=self.up_concat3(conv3,up4)
        
        up2=self.up_concat2(conv2,up3)
        
        up1=self.up_concat1(conv1,up2)
        
        final=self.final_1(up1)
        final=final + inputs
        final = self.final_2(final)
        return final