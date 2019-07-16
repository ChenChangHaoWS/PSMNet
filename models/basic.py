from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

########
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),      # convbn和convbn_3d的定义在submodule.py中，类中是可以调用外部函数的
                                     nn.ReLU(inplace=True),     # inplace覆盖运算，节省内存：https://www.cnblogs.com/wanghui-garcia/p/10642665.html
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)
 
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_(), volatile= not self.training).cuda()
# cost变量是cost volume的容器，torch.FloatTensor(**).zero_()类似于np.zeros()，用于生成**尺寸的的容器。此处是batch * channel * disparity * height * width。violated判断是否需要求梯度https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/

        for i in range(self.maxdisp/4): #因为feature_extraction后输出的图像尺寸长宽都缩小了1/4，所以现在的视差1就是真实的4
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:] # batch * channel * disparity * height * width和batch * channel * height * width
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
         # cost输出保持了1/4*D X (1/4*H X 1/4*W) X 64; 在W中，被剪切的部分用零补足
        cost = cost.contiguous()
# 防止cost volume因运算过程中其他量的改变（如refimg_fea等）而发生变化，起到重新拷贝赋予新的内存空间的作用：https://blog.csdn.net/gdymind/article/details/82662502
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.upsample(cost, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost = torch.squeeze(cost,1)
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)

        return pred
