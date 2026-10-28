import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class TrueEdgeEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
       
        self.edge_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self._init_fixed_filters() 
        
        for param in self.edge_conv.parameters():
            param.requires_grad = False
        
        self.post_conv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
    def _init_fixed_filters(self):
       
        kernel_set = [
            # Sobel水平
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            # Sobel垂直
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            # Prewitt水平
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
            # Prewitt垂直
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            # Laplacian
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            # 45度方向
            [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
            # 135度方向
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            # 各向同性Sobel
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        ]
        
        weights = torch.zeros(8, 1, 3, 3)
        for i in range(8):
            kernel = torch.tensor(kernel_set[i], dtype=torch.float32)
            kernel = kernel / torch.sum(torch.abs(kernel))  
            weights[i, 0, :, :] = kernel
            
        self.edge_conv.weight.data = weights

    def forward(self, x):
      
        edge_maps = self.edge_conv(x)  # [B,8,H,W]
        edge_strength = torch.abs(edge_maps)
        out = self.post_conv(edge_strength)  # [B,32,H/4,W/4]
        return out

class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(single_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class GSA(nn.Module):
   
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction_ratio
        self.channel_compression = nn.Conv2d(in_channels, in_channels//reduction_ratio, 1)
        self.query = nn.Conv2d(in_channels//reduction_ratio, 1, kernel_size=1)
        self.key = nn.Conv2d(in_channels//reduction_ratio, 1, kernel_size=1)
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        compressed = self.channel_compression(x)
        query = self.query(compressed)  
        key = self.key(compressed)      
        attention = torch.cat([query, key], dim=1)  
        attention = self.attention_conv(attention)  
        attention = F.leaky_relu(attention, 0.2)

        h, w = attention.shape[2:]
        attention = F.softmax(attention.view(-1, h*w), dim=1).view(-1, 1, h, w)
        
        out = x * attention
        out = self.transform(out)
        out = self.bn(out + residual)
        return out

class LSGR(nn.Module):
    def __init__(self, p2=4, nIn=32, nOut=32, add=True):
        super(LSGR, self).__init__()
        self.p2 = p2
        self.N = nIn // 4
        self.add = add

        self.graph0 = graph(p2, nIn, self.N)

        self.conv31 = nn.Sequential(
            nn.Conv2d(nOut, nOut, kernel_size=1, stride=1),
            nn.BatchNorm2d(nOut),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, C, H, W = x.shape
        h, w = H // self.p2, W // self.p2
        L = h * w

        fs = torch.zeros((b, self.p2 ** 2, C, h, w)).cuda()
        for i in range(1, self.p2 + 1):
            for j in range(1, self.p2 + 1):
                fs[:, i * j - 1, :, :, :] = x[:, :, (i - 1) * h: i * h, (j - 1) * w: j * w]
        fs = fs.view(b * self.p2 ** 2, C, h, w)

        fs6 = self.graph0(fs, x)

        out = torch.zeros_like(x)
        for i in range(1, self.p2 + 1):
            for j in range(1, self.p2 + 1):
                out[:, :, (i - 1) * h: i * h, (j - 1) * w: j * w] = fs6[:, i * j - 1, :, :, :]

        out = self.conv31(out)

        if self.add:
            out = out + x

        return out


class graph(nn.Module):
    def __init__(self, p2=4, nIn=64, N=16):
        super(graph, self).__init__()
        self.p2 = p2 
        self.N = N

        self.conv30 = nn.Sequential(
            nn.Conv2d(nIn, self.N, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True)
        )

        self.conv10 = nn.Sequential(
            nn.Conv1d(nIn, nIn, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.conv11 = nn.Sequential(
            nn.Conv1d(self.N, self.N, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.adaptivemax = nn.AdaptiveAvgPool2d((self.p2, self.p2))  

        self.conv12 = nn.Sequential(
            nn.Conv1d(p2 ** 2, p2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(p2, p2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(p2, p2 ** 2, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def ADP_weight(self, x):
        b, C, H, W = x.shape
        fg = self.adaptivemax(x)
        fg = fg.view(b, C, -1) 

       
        if fg.size(2) != self.p2 ** 2:
            raise ValueError(f"The number of pooled features ({fg.size(2)}) is not a multiple of {self.p2 ** 2}")

        fg1 = fg.view(b, C, self.p2, self.p2)
        fg1 = torch.transpose(fg1, 1, 3).contiguous().view(b, self.p2 ** 2, C)
        fg2 = self.conv12(fg1)
        fg3 = fg2.unsqueeze(-1).unsqueeze(-1)
        return fg3

    def graph_convolution(self, fs, x):
        b, C, H, W = x.shape
        h, w = H // self.p2, W // self.p2
        L = h * w

        B = self.conv30(fs)
        B1 = B.view(-1, self.N, L)

        fs1 = fs.view(-1, C, L)
        fs1 = torch.transpose(fs1, 1, 2)

        fs2 = torch.bmm(B1, fs1)
        fs3 = self.conv11(fs2)

        fs5 = self.conv10(torch.transpose(fs3, 1, 2))
        fs6 = torch.bmm(torch.transpose(B1, 1, 2), torch.transpose(fs5, 1, 2))
        fs6 = torch.transpose(fs6, 1, 2)
        fs6 = fs6.view(b, self.p2 ** 2, C, h, w)
        return fs6

    def forward(self, fs, x):
        fs6 = self.graph_convolution(fs, x)
        weight = self.ADP_weight(x)
        out = weight * fs6
        return out


class HSGR(nn.Module):
    def __init__(self, inch, N=16):
        super(HSGR, self).__init__()
        self.N = N

        self.conv11 = single_conv(inch, inch, 3, 1)
        self.conv12 = nn.Sequential(
            nn.Conv2d(inch, inch, 3, 1, dilation=3, padding=3),
            nn.BatchNorm2d(inch),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(inch, inch, 3, 1, dilation=5, padding=5),
            nn.BatchNorm2d(inch),
            nn.ReLU(inplace=True)
        )

        self.convN = nn.Conv2d(inch, N, 1, 1, padding=0, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv1d(3 * N, N, 1, 1, bias=False),
            nn.Conv1d(N, 3 * N, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv21 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv31 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv33 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # print(f"HSGR input shape: {x.shape}")  

        x10 = self.conv11(x)
        x20 = self.conv12(x)
        x30 = self.conv13(x)

        x1 = x10.view(B, C, H * W)
        x2 = x20.view(B, C, H * W)
        x3 = x30.view(B, C, H * W)

        w = self.convN(x).view(B, self.N, H * W)  

        xN1 = self.conv21(torch.bmm(w, torch.transpose(x1, 1, 2)))  
        xN2 = self.conv22(torch.bmm(w, torch.transpose(x2, 1, 2)))  
        xN3 = self.conv23(torch.bmm(w, torch.transpose(x3, 1, 2)))  

        xN = torch.cat([xN1, xN2, xN3], dim=1) 
        xNout = self.conv2(xN)  

        y1 = self.conv31(xNout[:, :self.N, :])  
        y2 = self.conv32(xNout[:, self.N: (self.N * 2), :])  
        y3 = self.conv33(xNout[:, (self.N * 2): (self.N * 3), :])  

        out1 = torch.bmm(torch.transpose(y1, 1, 2), w).view(B, C, H, W)
        out2 = torch.bmm(torch.transpose(y2, 1, 2), w).view(B, C, H, W)
        out3 = torch.bmm(torch.transpose(y3, 1, 2), w).view(B, C, H, W)

        out = out1 + out2 + out3
        # print(f"HSGR output shape: {out.shape}") 
        return out
