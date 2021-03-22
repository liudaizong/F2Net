# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:01:14 2018

@author: carri
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
affine_par = True
#区别于siamese_model_concat的地方就是采用的最标准的deeplab_v3的基础网络，然后加上了非对称的分支

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
        stride=stride, bias=False)

def torch_make_gauss(input_size, ct, scale=15):
    gauss_batch = []
    for i in range(ct.size(0)):
        h, w = input_size
        # if torch.sum(mask)==0:
        #     gauss = torch.ones((h,w)).cuda()
        #     gauss_batch.append(gauss)
        #     continue
        center_x, center_y = ct[i]
        x = torch.arange(0., w, 1)
        y = torch.arange(0., h, 1).unsqueeze(-1)
        gauss = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / 2.0 / scale / scale).cuda()
        gauss_batch.append(gauss)
    gauss_batch = torch.stack(gauss_batch,dim=0).unsqueeze(dim=1).cuda()
    return gauss_batch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv= nn.Conv2d(2048, depth, 1,1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0], dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1], dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2], dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d( depth*5, 256, kernel_size=3, padding=1 )  #512 1x1Conv
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        #for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)#classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)
        

    def forward(self, x):
        #out = self.conv2d_list[0](x)
        #mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size=x.shape[2:]
        image_features=self.mean(x)
        image_features=self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features=F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0) 
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1) 
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2) 
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3) 
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        #for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)
        
        return out
  


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(ASPP, [ 6, 12, 18], [6, 12, 18], 512)
        self.main_classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        self.softmax = nn.Sigmoid()#nn.Softmax()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        fea = self.layer5(x4)
        x = self.main_classifier(fea)
        #print("before upsample, tensor size:", x.size())
        x = F.upsample(x, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x = self.softmax(x)
        return fea, x, [x4, x3, x2, x1]

class Upsample_unit(nn.Module): 

    def __init__(self, ind, in_planes, chl_num, scale):
        super(Upsample_unit, self).__init__()

        self.u_skip = conv1x1(in_planes, chl_num)
        self.bn1 = nn.BatchNorm2d(chl_num)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_conv = conv1x1(chl_num, chl_num)
            self.bn2 = nn.BatchNorm2d(chl_num)

        if self.ind == 3:
            self.cross_conv = conv1x1(chl_num, 64)
            self.bn5 = nn.BatchNorm2d(64)
            self.relu_cross = nn.ReLU(inplace=True)

        self.scale = scale

    def forward(self, x, up_x):
        out = self.bn1(self.u_skip(x))

        if self.ind > 0:
            up_x = F.interpolate(up_x, self.scale, mode='bilinear')
            up_x = self.bn2(self.up_conv(up_x))
            out += up_x 
        out = self.relu(out)

        cross_conv = None
        if self.ind == 3:
            cross_conv = self.relu_cross(self.bn5(self.cross_conv(out)))

        return out, cross_conv


class Upsample_module(nn.Module):

    def __init__(self, chl_num=256):
        super(Upsample_module, self).__init__()
        self.in_planes = [2048, 1024, 512, 256]
        self.out_planes = [256, 256, 256, 256]
        self.scale = [(60, 60), (60, 60), (60, 60), (119, 119)]

        self.up1 = Upsample_unit(0, self.in_planes[0], self.out_planes[0], self.scale[0])
        self.up2 = Upsample_unit(1, self.in_planes[1], self.out_planes[1], self.scale[1])
        self.up3 = Upsample_unit(2, self.in_planes[2], self.out_planes[2], self.scale[2])    
        self.up4 = Upsample_unit(3, self.in_planes[3], self.out_planes[3], self.scale[3])


    def forward(self, layers):
        x4, x3, x2, x1 = layers
        out1, _ = self.up1(x4, None)
        out2, _ = self.up2(x3, out1)
        out3, _ = self.up3(x2, out2)
        out4, cross_conv = self.up4(x1, out3)

        return cross_conv

class CoattentionModel(nn.Module):
    def  __init__(self, block, layers, num_classes, all_channel=256, all_dim=60*60):	#473./8=60	
        super(CoattentionModel, self).__init__()
        self.encoder = ResNet(block, layers, num_classes)

        self.upsample = Upsample_module()
        self.fc = nn.Sequential(
                      nn.Conv2d(64, 256,
                                kernel_size=3, padding=1, bias=True),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(256, 1,
                                kernel_size=1, stride=1,
                                padding=1 // 2, bias=True))
        self.fc[-1].bias.data.fill_(-2.19)

        self.motion_conv_w = nn.Conv2d(64+1, 1, kernel_size=1, stride=1, bias=True) #all_channel
        self.motion_conv_b = nn.Conv2d(64+1, 1, kernel_size=1, stride=1, bias=True)

        self.channel_attention_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_conv1 = nn.Conv2d(all_channel, all_channel//16, 1, bias=False)
        self.channel_attention_bn = nn.BatchNorm2d(all_channel//16)
        self.channel_attention_conv2 = nn.Conv2d(all_channel//16, 3 * all_channel, 1, bias=False)

        self.spatial_attention_conv = nn.Conv2d(all_channel, 3, 1, bias=False)
        

        self.cls = nn.Sequential(
            nn.Conv2d(all_channel, all_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(all_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channel, num_classes, kernel_size=1, stride=1, padding=0)
        )

        # self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        # self.channel = all_channel
        # self.dim = all_dim
        # self.gate = nn.Conv2d(all_channel, 1, kernel_size  = 1, bias = False)
        # self.gate_s = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
        # self.conv2 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
        # self.bn1 = nn.BatchNorm2d(all_channel)
        # self.bn2 = nn.BatchNorm2d(all_channel)
        # self.prelu = nn.ReLU(inplace=True)
        # self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        # self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()
        self.D = all_channel
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    
    
		
    def forward(self, input1, previous, input2, ct, input_guass): #注意input2 可以是多帧图像
        #input1_att, input2_att = self.coattention(input1, input2) 
        input_size = input1.size()[2:]
        ref_features, temp, ref_layers = self.encoder(input1)
        # pre_features, _, _ = self.encoder(previous)
        curr_features, temp, curr_layers = self.encoder(input2)

        batch, channel, h, w = curr_features.shape
        M = h * w

        #center based guass
        gauss = torch_make_gauss(input_size, ct)
        gauss = F.interpolate(gauss,(h,w),mode='bilinear',align_corners=False)
        # que_key_gauss = torch.cat([curr_features,gauss],dim=1)
        # motion_w = self.motion_conv_w(que_key_gauss)
        # motion_w = torch.sigmoid(motion_w)
        # motion_b = self.motion_conv_b(que_key_gauss)
        # motion_b = torch.sigmoid(motion_b)
        # motion = motion_w*gauss+motion_b # b,1,h,w
        # motion = motion.view(batch,1,h*w)
        motion = gauss.view(batch, 1, h*w)

        #heatmap for ct prediction
        # ref_upheat = self.upsample(ref_layers)
        curr_upheat = self.upsample(curr_layers)
        # ref_upheat = ref_upheat.view(batch, 64, 119*119)
        # curr_upheat = curr_upheat.view(batch, 64, 119*119)
        # curr_ref = torch.matmul(curr_upheat.permute(0, 2, 1), ref_upheat)
        # curr_ref_key = F.softmax((channel ** -.5) * curr_ref, dim=-1)
        # input1_guass = input1_guass.view(batch, 1, 119*119)
        # curr_ref_key = curr_ref_key * input1_guass
        # curr_ref_value = torch.matmul(curr_ref_key, ref_upheat.permute(0, 2, 1)).permute(0, 2, 1)
        # upheat = torch.cat([curr_ref_value.view(batch, 64, 119, 119), curr_upheat.view(batch, 64, 119, 119)], dim=1)
        upheat = torch.cat([curr_upheat, input_guass], dim=1)
        # heat = self.fc(upheat)
        motion_w = self.motion_conv_w(upheat)
        motion_w = torch.sigmoid(motion_w)
        motion_b = self.motion_conv_b(upheat)
        motion_b = torch.sigmoid(motion_b)
        heat1 = motion_w*input_guass+motion_b # b,1,h,w
        heat2 = self.fc(curr_upheat)
        heat = heat1 + heat2

        #non-local
        ref_features = ref_features.view(batch, channel, M)#.permute(0, 2, 1)
        # pre_features = pre_features.view(batch, channel, M)
        curr_features = curr_features.view(batch, channel, M)

        p_0 = torch.matmul(ref_features.permute(0, 2, 1), curr_features)
        p_0 = F.softmax((channel ** -.5) * p_0, dim=-1)
        # p_0 = p_0.permute(0, 2, 1)
        # p_0 = p_0 * motion
        # p_0 = p_0.permute(0, 2, 1)
        p_1 = torch.matmul(curr_features.permute(0, 2, 1), curr_features)
        p_1 = F.softmax((channel ** -.5) * p_1, dim=-1)
        # p_1 = p_1.permute(0, 2, 1)
        # p_1 = p_1 * motion
        # p_1 = p_1.permute(0, 2, 1)
        # p_2 = torch.matmul(curr_features.permute(0, 2, 1), pre_features)
        # p_2 = F.softmax((channel ** -.5) * p_2, dim=-1)
        # p_2 = p_2.permute(0, 2, 1)
        # p_2 = p_2 * motion
        # p_2 = p_2.permute(0, 2, 1)
        feats_0 = torch.matmul(p_0, curr_features.permute(0, 2, 1)).permute(0, 2, 1)
        feats_1 = torch.matmul(p_1, curr_features.permute(0, 2, 1)).permute(0, 2, 1)
        # feats_2 = torch.matmul(p_2, pre_features.permute(0, 2, 1)).permute(0, 2, 1)
        
        #channel-attention
        x = feats_0 + feats_1 + curr_features
        x = x.view(batch, channel, h, w)
        x = self.channel_attention_pool(x)
        x = F.relu(self.channel_attention_bn(self.channel_attention_conv1(x)))
        x = self.channel_attention_conv2(x)
        x = torch.unsqueeze(x, 1).view(-1, 3, self.D, 1, 1)
        x = F.softmax(x, 1)
        x1 = feats_0.view(batch, channel, h, w) * x[:, 0, :, :, :]
        x2 = feats_1.view(batch, channel, h, w) * x[:, 1, :, :, :]
        x3 = curr_features.view(batch, channel, h, w) * x[:, 2, :, :, :]
        x  = x1 + x2 + x3
        
        #spatial-attention
        # x = feats_0 + feats_1 + curr_features
        # x = x.view(batch, channel, h, w)
        # x = self.spatial_attention_conv(x)
        # x = F.softmax(x, 1)
        # x1 = feats_0.view(batch, channel, h, w) * x[:, 0, :, :].view(batch, 1, h, w)
        # x2 = feats_1.view(batch, channel, h, w) * x[:, 1, :, :].view(batch, 1, h, w)
        # x3 = curr_features.view(batch, channel, h, w) * x[:, 2, :, :].view(batch, 1, h, w)
        # x  = x1 + x2 + x3
        
        #channel-spatial-attention
        # x = feats_0 + feats_1 + curr_features
        # x = x.view(batch, channel, h, w)
        # x = self.channel_attention_pool(x)
        # x = F.relu(self.channel_attention_bn(self.channel_attention_conv1(x)))
        # x = self.channel_attention_conv2(x)
        # x = torch.unsqueeze(x, 1).view(-1, 3, self.D, 1, 1)
        # x = F.softmax(x, 1)
        # x1 = feats_0.view(batch, channel, h, w) * x[:, 0, :, :, :]
        # x2 = feats_1.view(batch, channel, h, w) * x[:, 1, :, :, :]
        # x3 = curr_features.view(batch, channel, h, w) * x[:, 2, :, :, :]
        # x  = x1 + x2 + x3
        # x = x.view(batch, channel, h, w)
        # x = self.spatial_attention_conv(x)
        # x = F.softmax(x, 1)
        # x1 = x1.view(batch, channel, h, w) * x[:, 0, :, :].view(batch, 1, h, w)
        # x2 = x2.view(batch, channel, h, w) * x[:, 1, :, :].view(batch, 1, h, w)
        # x3 = x3.view(batch, channel, h, w) * x[:, 2, :, :].view(batch, 1, h, w)
        # x  = x1 + x2 + x3
        
        # x = feats_1
        # x = x.view(batch, channel, h, w)
        # x = torch.cat([feats_0, feats_1, curr_features], dim=1).view(batch, 3 * channel, h, w)
        
        pred = self.cls(x)
        pred = F.upsample(pred, input_size, mode='bilinear')
        pred = self.softmax(pred)
    
        return pred, temp, heat #shape: NxCx	
    

def Res_Deeplab(num_classes=2):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes-1)
    return model

def CoattentionNet(num_classes=2):
    model = CoattentionModel(Bottleneck,[3, 4, 23, 3], num_classes-1)
	
    return model
