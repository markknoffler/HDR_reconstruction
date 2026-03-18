import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.checkpoint import checkpoint


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,      
            groups=in_channels,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)

class DAB_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca_fc = nn.Linear(channels, channels)
        self.pa_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        B,C,H,W = x.size()
        ca = F.adaptive_avg_pool2d(x, output_size=1)
        ca = ca.view(B, C)
        ca = self.ca_fc(ca)
        ca = torch.sigmoid(ca).view(B, C, 1, 1)
        CA = x * ca

        s_avg = x.mean(dim=1, keepdim=True)
        s_score = torch.sigmoid(s_avg)
        SA = x * s_score

        p_score = torch.sigmoid(self.pa_conv(x))
        PA = x * p_score
        
        out = CA + SA + PA
        return out

class PFF_block_3(nn.Module):
    def __init__(self, ch_pre, ch_curr, ch_nex, dab_block_cls):
        super().__init__()
        self.down_dw = DepthwiseConv2d(ch_pre, kernel_size=2, stride=2)
        self.down_1x1 = nn.Conv2d(ch_pre, ch_curr, kernel_size=1, stride =1, padding=0)

        self.up_1x1 = nn.Conv2d(ch_nex, ch_curr, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        concat_channels =  ch_curr * 2 
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_curr, kernel_size=1, stride=1, padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=True)

    def forward(self, pre, curr, next):
        down = self.down_dw(pre)
        down = self.down_1x1(down)

        up = self.up_1x1(next)
        up = self.upsample(up)

        mul_low = down * curr
        mul_high = up * curr

        concat = torch.cat([mul_low, mul_high], dim=1)

        att = self.dab(concat)

        x = self.out_1x1(att)
        x = self.act(x)

        return x

class PFF_block_pre(nn.Module):
    def __init__(self, ch_curr, ch_next):
        super().__init__()
        self.up_1x1 = nn.Conv2d(ch_next, ch_curr, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        concat_channels = ch_curr * 2
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_curr, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=True)

    def forward(self, curr, next):
        up = self.up_1x1(next)
        up = self.upsample(up)

        mul = up * curr

        concat = torch.cat([curr, mul], dim=1)

        att = self.dab(concat)
        x = self.out_1x1(att)
        x = self.act(x)

        return x

class PFF_block_next(nn.Module):
    def __init__(self, ch_curr, ch_prev):
        super().__init__()
        self.down_dw = DepthwiseConv2d(ch_prev, kernel_size=2, stride=2)
        self.down_1x1 = nn.Conv2d(ch_prev, ch_curr, kernel_size=1, stride=1, padding=0)

        concat_channels = 2 * ch_curr
        self.dab = DAB_block(concat_channels)

        self.out_1x1 = nn.Conv2d(concat_channels, ch_curr, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=True)

    def forward(self, curr, prev):
        down = self.down_dw(prev)
        down = self.down_1x1(down)

        mul = down * curr

        concat = torch.cat([curr, mul], dim=1)

        att = self.dab(concat)
        x = self.out_1x1(att)
        x = self.act(x)

        return x

class semantic_codebook_encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=4):
        super(semantic_codebook_encoder, self).__init__()

        self.block1 = self._make_block(in_channels, base_channels)
        self.block2 = self._make_block(base_channels, base_channels*2)
        self.block3 = self._make_block(base_channels*2, base_channels*4)
        self.block4 = self._make_block(base_channels*4, base_channels*8)

        self.seg_feat1, self.seg_feat2, self.seg_feat3, self.seg_feat4 = semantic_map_encoder(block1, block2, block3, block4)

        self.comb_block1 = self.block1 + self.seg_feat1
        self.comb_block2 = self.block2 + self.seg_feat2
        self.comb_block3 = self.block3 + self.seg_feat3
        self.comb_block4 = self.block4 + self.seg_feat4

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, ldr_input, segmap):
        cnn_feat_layer1 = self.block1(ldr_input)
        cnn_feat_layer2 = self.block2(cnn_feat)
        cnn_feat_layer3 = self.block3(cnn_feat)
        cnn_feat_layer4 = self.block4(cnn_feat)

        seg_feat1, seg_feat2, seg_feat3, seg_feat4 = self.semantic_map_encoder(cnn_feat_layer1, cnn_feat_layer2, cnn_feat_layer3, cnn_feat_layer4)

        comb_block1 = cnn_feat_layer1 + seg_feat1
        comb_block2 = cnn_feat_layer2 + seg_feat2
        comb_block3 = cnn_feat_layer3 + seg_feat3
        comb_block4 = cnn_feat_layer4 + seg_feat4

        return comb_block1, comb_block2, comb_block3, comb_block4 

class semantic_map_encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=4):
        super(semantic_map_encoder, self).__init__()

        self.block1 = self._make_block(in_channels, base_channels)
        self.block2 = self._make_block(base_channels, base_channels*2)
        self.block3 = self._make_block(base_channels*2, base_channels*4)
        self.block4 = self._make_block(base_channels*4, base_channels*8)

#        self.pff_block_1 = PFF_block_pre(base_channels, base_channels*2)
#        self.pff_block_2 = PFF_block_3(base_channels, base_channels*2, base_channels*4)
#        self.pff_block_3 = PFF_block_3(base_channels*2, base_channels*4, base_channels*8)
#        self.pff_block_4 = PFF_block_next(base_channels*4, base_channels*8)
#

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, seg_map):
        cnn_feat_layer1 = self.block1(seg_map)
        cnn_feat_layer2 = self.block2(cnn_feat_layer1)
        cnn_feat_layer3 = self.block3(cnn_feat_layer2)
        cnn_feat_layer4 = self.block4(cnn_feat_layer3)

#        pff1 = self.pff_block_1(cnn_feat_layer1, cnn_feat_layer2)
#        pff2 = self.pff_block_2(cnn_feat_layer1, cnn_feat_layer2, cnn_feat_layer3)
#        pff3 = self.pff_block_3(cnn_feat_layer2, cnn_feat_layer3, cnn_feat_layer4)
#        pff4 = self.pff_block_4(cnn_feat_layer3, cnn_feat_layer4)
#
        return cnn_feat_layer1, cnn_feat_layer2, cnn_feat_layer3, cnn_feat_layer4    


#big question why do need the pff at all? we can directly inject the segmap features into the encoder
#but you see in our model we can use pff block although we dont need to focus on small and large details at the same time
#we can use it by making the pff block see through the different encoders at the same time instead of a single encoder
#inject the DAB and the pff blocks to see all the 3 encoders at once during and during only the decoding
