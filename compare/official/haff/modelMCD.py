import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaptiveModel import DCMModle
from .AdaptiveModel import Dynamic_conv2d
from .AdaptiveModel import attention2d
from .PostionAM import PAM
class ADVNets(nn.Module):
    def __init__(self,input_nbr, label_nbr):
        super(ADVNets, self).__init__()
        self.input_nbr = input_nbr
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.maxpool1 =  nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

  #*********************************************************************


        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        ################################################
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.PAM1 = PAM(32)
        self.DCM11 = DCMModle(32, 8, filter_size=1, fusion=True)
        self.DCM12 = DCMModle(32, 8, filter_size=3, fusion=True)
        self.DCM13 = DCMModle(32, 8, filter_size=5, fusion=True)
        self.DCM14 = DCMModle(32, 8, filter_size=7, fusion=True)
        self.PAM2 = PAM(64)
        self.DCM21 = DCMModle(64, 16, filter_size=1, fusion=True)
        self.DCM22 = DCMModle(64, 16, filter_size=3, fusion=True)
        self.DCM23 = DCMModle(64, 16, filter_size=5, fusion=True)
        self.DCM24 = DCMModle(64, 16, filter_size=7, fusion=True)
        self.PAM3 = PAM(128)
        self.DCM31 = DCMModle(128, 32, filter_size=1, fusion=True)
        self.DCM32 = DCMModle(128, 32, filter_size=3, fusion=True)
        self.DCM33 = DCMModle(128, 32, filter_size=5, fusion=True)
        self.DCM34 = DCMModle(128, 32, filter_size=7, fusion=True)
        self.PAM4 = PAM(256)
        self.DCM41 = DCMModle(256, 64, filter_size=1, fusion=True)
        self.DCM42 = DCMModle(256, 64, filter_size=3, fusion=True)
        self.DCM43 = DCMModle(256, 64, filter_size=5, fusion=True)
        self.DCM44 = DCMModle(256, 64, filter_size=7, fusion=True)
        #反卷积
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.Dconv1 = nn.Sequential(
            nn.ConvTranspose2d(160, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, label_nbr, kernel_size=3, padding=1)
        )
        self.Dconv2 = nn.Sequential(
            nn.ConvTranspose2d(192, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.Dconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        )

        self.Dconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.Dconv3_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Dconv3_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, label_nbr, kernel_size=3, padding=1)
        )
        self.Dconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.Dconv4_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Dconv4_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.Dconv4_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, label_nbr, kernel_size=3, padding=1)
        )
        self.Dynamic = Dynamic_conv2d(16, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4)
        self.sm = nn.LogSoftmax(dim=1)
        self.last = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, label_nbr, kernel_size=3, padding=1),
        )
        self.attention = attention2d(4,4)
    def forward(self, x1, x2):
        # Stage 1
        img1 = self.conv1(x1)
        img2 = self.conv1(x2)
        img1_1 = self.maxpool1(img1)
        img2_1 = self.maxpool1(img2)
        img1 = torch.cat((img1_1, img2_1), 1)
        #print(img1.shape)
        img1 = self.PAM1(img1)
        DCM1_1 = self.DCM11(img1)
        DCM1_2 = self.DCM12(img1)
        DCM1_3 = self.DCM13(img1)
        DCM1_4 = self.DCM14(img1)
        img111 =torch.cat([DCM1_1,DCM1_2,DCM1_3,DCM1_4],1)
#*********************************************最后一层********************
       # print(img11.shape)
        img1_2 = self.conv2(img1_1)
        img2_2 = self.conv2(img2_1)
        img1_3 = self.maxpool2(img1_2)
        img2_3 = self.maxpool2(img2_2)
        img2 = torch.cat((img1_3, img2_3), 1)
       # print(img2.shape)
        img2 = self.PAM2(img2)
        DCM2_1 = self.DCM21(img2)
        DCM2_2 = self.DCM22(img2)
        DCM2_3 = self.DCM23(img2)
        DCM2_4 = self.DCM24(img2)
        img222 = torch.cat([DCM2_1, DCM2_2, DCM2_3, DCM2_4], 1)
        #print(img22.shape)
        img1_4 = self.conv3(img1_3)
        img2_4 = self.conv3(img2_3)
        img1_5 = self.maxpool3(img1_4)
        img2_5 = self.maxpool3(img2_4)
        img3 = torch.cat((img1_5, img2_5), 1)
        img3 = self.PAM3(img3)
        DCM3_1 = self.DCM31(img3)
        DCM3_2 = self.DCM32(img3)
        DCM3_3 = self.DCM33(img3)
        DCM3_4 = self.DCM34(img3)
        img333 = torch.cat([DCM3_1, DCM3_2, DCM3_3, DCM3_4], 1)
        #print(img33.shape)
        img1_6 = self.conv4(img1_5)
        img2_6 = self.conv4(img2_5)
        img1_7 = self.maxpool3(img1_6)
        img2_7 = self.maxpool3(img2_6)
        img4 = torch.cat((img1_7, img2_7), 1)
        img4 = self.PAM4(img4)
        DCM4_1 = self.DCM41(img4)
        DCM4_2 = self.DCM42(img4)
        DCM4_3 = self.DCM43(img4)
        DCM4_4 = self.DCM44(img4)
        img444 = torch.cat([DCM4_1, DCM4_2, DCM4_3, DCM4_4], 1)
        img44_1 = self.Dconv4(img444) ###img444
        img44_2 = self.Dconv4_1(img44_1)
        img44_3 = self.Dconv4_2(img44_2)
        Img1 = self.Dconv4_3(img44_3)
#*****************三层************************
        img334 = torch.cat([img44_1,img333], 1)###img444
        img33_1 = self.Dconv3(img334)
        img33_2 = self.Dconv3_1(img33_1)
        Img2 = self.Dconv3_2(img33_2)
#**********************************************
        img223 = torch.cat([img33_1, img222], 1)###img444
        img22_1 = self.Dconv2(img223)
        Img3 = self.Dconv2_1(img22_1)

 # **********************************************
        img112 = torch.cat([img22_1, img111], 1)###img444
        Img4 = self.Dconv1(img112)
        return Img4,Img3,Img2,Img1

if __name__ == '__main__':
    net = ADVNets(input_nbr=3, label_nbr=1)
    image1 = torch.randn(3, 3, 128, 128)
    image2 = torch.randn(3, 3,128, 128)
    out1,_,_,_ = net(image1,image2)
    print(out1.shape)




