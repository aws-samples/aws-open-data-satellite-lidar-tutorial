# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

""" Originally from CRESI project. We modified the code to accomodate
different number of input channels, e.g. 5-channel RGB+LIDAR input images.
"""

import math
import torch
from torch import nn


################################################################################
# Function to be called from external
def get_modified_resnet_unet(in_channels=3, num_classes=8, logits=False):
    """ Get a modified ResNet-Unet model with customized input channel numbers
    and output classes numbers.
    For example, we can set in_channels=3 and input RGB 3-channel images.
    On the other hand, we can set in_channels=5 if we want to input both RGB
    and 2-channel LIDAR data (elevation + intensity).
    As for the output, we can do either binary classification (num_classes=2)
    or multi-class classification. In this case, the default num_classes=8 is
    used for SpaceNet route and speed estimation.

    By default, ResNet34 is used. It can be changed to ResNet50 or ResNet101.
    """
    class Modified_ResNetUnet(ResNetUnet):
        def __init__(self):
            super().__init__(in_channels=in_channels, num_classes=num_classes,
                             encoder_name='resnet34', logits=logits)
    return Modified_ResNetUnet


################################################################################
# Define ResNet setups
def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

encoder_params = {
    'resnet34': {'filters': [64, 64, 128, 256, 512],
                 'init_op': resnet34},
    'resnet50': {'filters': [64, 256, 512, 1024, 2048],
                 'init_op': resnet50},
    'resnet101': {'filters': [64, 256, 512, 1024, 2048],
                  'init_op': resnet101}
}

def conv3x3(in_filters, out_filters, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            residual = self.downsample(x)
        y += residual
        y = self.relu(y)
        return y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            residual = self.downsample(x)
        y += residual
        y = self.relu(y)
        return y

class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y


################################################################################
# Define Unet encoder-decoder structure
class ConvBottleneck(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class UnetDecoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming He normal initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    # def initialize_encoder(self, model, model_url):
    #     pretrained_dict = model_zoo.load_url(model_url)
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model.load_state_dict(pretrained_dict)

    @property
    def first_layer_params_name(self):
        return 'conv1'

class EncoderDecoder(AbstractModel):
    def __init__(self, in_channels, num_classes, encoder_name='resnet34', logits=False):
        super().__init__()
        self.filters = encoder_params[encoder_name]['filters']
        self.in_channels = in_channels
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(f*2, f) for
                                          f in reversed(self.filters[:-1])])
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for
                                             idx in range(1, len(self.filters))])
        self.last_upsample = UnetDecoderBlock(self.filters[0], self.filters[0]//2)
        self.final = self.make_final_classifier(self.filters[0]//2, num_classes)
        self.logits = logits
        
        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](in_channels=in_channels)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx)
                                             for idx in range(len(self.filters))])
        
    def forward(self, x):
        enc_results = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if idx < len(self.encoder_stages) - 1:
                enc_results.append(x.clone())
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx])
        x = self.last_upsample(x)
        x = self.final(x)
        if self.logits:
            return x
        else:
            return torch.sigmoid(x)
    
    def get_decoder(self, layer):
        return UnetDecoderBlock(self.filters[layer], self.filters[max(layer-1, 0)])
    
    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(nn.Conv2d(in_filters, num_classes, 3, padding=1))
    
    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])
    
    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + \
                 [self.bottlenecks, self.decoder_stages, self.final]
        return _get_layers_params(layers)

def _get_layers_params(layers):
    return sum((list(l.paramters()) for l in layers), [])

def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


################################################################################
# Put together ResNet and Unet.
class ResNetUnet(EncoderDecoder):
    def __init__(self, in_channels, num_classes, encoder_name, logits=False):
        super().__init__(in_channels, num_classes, encoder_name, logits=logits)
    
    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        elif layer == 1:
            return nn.Sequential(encoder.maxpool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4
