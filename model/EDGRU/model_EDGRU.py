import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *  


class EDGRU(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(EDGRU, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Main Encoder
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv5 = conv_block(filters[3], filters[4])

        # Auxiliary Encoder with channel alignment
        self.edge_encoder = nn.Sequential(
            TrueEdgeEncoder(in_ch), 
            nn.Conv2d(32, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Feature Fusion Modules
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(filters[0]*2, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        self.deep_fusion = nn.Sequential(
            nn.Conv2d(filters[4]*2, filters[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()

        # Graph Reasoning Modules
        self.lsgr = LSGR(p2=4, nIn=filters[0], nOut=filters[0], add=False)
        self.hsgr = HSGR(inch=filters[4], N=16)

        # Attention Mechanisms
        self.graph_attention1 = GSA(filters[0])
        self.graph_attention2 = GSA(filters[1])
        self.graph_attention3 = GSA(filters[2])
        self.graph_attention4 = GSA(filters[3])
    def forward(self, x):
        # Main Encoder
        e1 = self.Conv1(x)
        
        # Auxiliary Encoder
        edge_feat = self.edge_encoder(x)
        edge_feat = F.interpolate(edge_feat, size=e1.shape[2:], mode='bilinear', align_corners=True)
        
        # Shallow Feature Fusion
        fused_e1 = self.edge_fusion(torch.cat([e1, edge_feat], dim=1))
        e1_lsgr = self.lsgr(fused_e1)
        
        e2 = self.Maxpool1(fused_e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        
        # Deep Feature Enhancement
        e5_hsgr = self.hsgr(e5)
        
        # Decoder with skip connections
        d5 = self.Up5(e5_hsgr)
        d5 = torch.cat([self.graph_attention4(e4), d5], dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat([self.graph_attention3(e3), d4], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat([self.graph_attention2(e2), d3], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat([self.graph_attention1(e1_lsgr), d2], dim=1)
        d2 = self.Up_conv2(d2)

        out = self.active(self.Conv(d2))
        return out