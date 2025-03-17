import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import clip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pywt


def mercator_project(latitude, longitude):
    """
    将经纬度Tensor转换为墨卡托投影坐标（支持批量计算）
    
    参数:
        latitude (Tensor):  纬度（单位：度），形状任意
        longitude (Tensor): 经度（单位：度），形状与latitude相同
        
    返回:
        tuple: (x, y) 投影后的坐标Tensor（单位：米），形状与输入一致
    """
    # 将经纬度转换为弧度
    lat_rad = torch.deg2rad(latitude)
    lon_rad = torch.deg2rad(longitude)
    
    # 地球半径（Web墨卡托标准参数，单位：米）
    R = 6378137
    
    # 计算墨卡托投影坐标
    x = R * lon_rad
    y = R * torch.log(torch.tan(torch.pi/4 + lat_rad/2))
    
    return x, y
def equirectangular_project(lat, lon):
    """
    等距圆柱投影，并对经度方向进行纬度缩放校正。
    
    参数:
    - lat: 纬度 (tensor)
    - lon: 经度 (tensor)
    
    返回:
    - lat_proj: 投影后的纬度
    - lon_proj: 投影后的经度
    """
    # 使用纬度的余弦值对经度进行缩放
    lat_rad = torch.deg2rad(lat)  # 将纬度从度转换为弧度
    lon_proj = lon * torch.cos(lat_rad)  # 对经度进行缩放
    lat_proj = lat  # 纬度保持不变
    
    return lat_proj, lon_proj
import torch.nn.utils.rnn as rnn_utils
# 定义BiLSTM特征提取器
class BiLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTMFeatureExtractor, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.bilstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x, mask = None, idx = -1):
        #x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.bilstm(x)  # lstm_out: (batch_size, seq_len, 2*hidden_size)

        #lengths = mask.sum(dim=1).cpu().numpy()
        
        #print(x)
        # 取最后一个时间步的输出作为航迹特征
        feature = lstm_out[:, idx, :]  # (batch_size, 2*hidden_size)
        return feature

import os.path
# 定义TTCSN模型
class MCLITA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MCLITA, self).__init__()

        hidden_size = 256
        # 在使用模型前确保设置了CUDA_LAUNCH_BLOCKING
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

                # 多尺度CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2)

        # 使用共享权重的BiLSTM
        self.feature_extractor = BiLSTMFeatureExtractor(198, hidden_size)
        # 分类器
        self.fc1 = nn.Linear(hidden_size*2, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    def remove_padding(self, x, padding_value=0):
        """去除输入中的填充部分"""
        if isinstance(x, torch.Tensor):
            mask = ~torch.all(x == padding_value, dim=-1)  # 找到非填充部分的掩码
            non_padding_lengths = mask.sum(dim=1)
            outputs = [x[i, :non_padding_lengths[i]] for i in range(x.size(0))]
            return outputs, mask
        else:
            raise ValueError("Unsupported input type. Expected a torch tensor.")
    def forward(self, x1, x2, dtime):
        # 将输入转置以适应Conv1D的要求 (batch_size, channels, seq_length)
        x1, x2 ,_= normalize_coordinates(x1, x2)
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)

        # 多尺度卷积
        conv1_out1 = torch.relu(self.conv1(x1))
        conv3_out1 = torch.relu(self.conv3(x1))
        conv5_out1 = torch.relu(self.conv5(x1))
        conv1_out2 = torch.relu(self.conv1(x2))
        conv3_out2 = torch.relu(self.conv3(x2))
        conv5_out2 = torch.relu(self.conv5(x2))

        # 合并不同尺度的特征
        x1 = torch.cat((x1,conv1_out1, conv3_out1, conv5_out1), dim=1).transpose(1, 2)
        x2 = torch.cat((x2,conv1_out2, conv3_out2, conv5_out2), dim=1).transpose(1, 2)

        h1 = self.feature_extractor(x1,-1)  # (batch_size, 2*hidden_size)
        h2 = self.feature_extractor(x2,0)  # (batch_size, 2*hidden_size)
        #print(h2.shape)
        # 计算L1范数距离
        d = torch.abs(h1 - h2)  # (batch_size, 2*hidden_size)

        # 分类器
        out = self.fc1(d)
        out = torch.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # (batch_size, 1)
        return out
    
    # 定义TTCSN模型
class DCMCLITA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DCMCLITA, self).__init__()

        hidden_size = 256
        # 在使用模型前确保设置了CUDA_LAUNCH_BLOCKING
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

                # 多尺度CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2)

        # 使用共享权重的BiLSTM
        self.feature_extractor = BiLSTMFeatureExtractor(198, hidden_size)
        # 分类器
        self.fc1 = nn.Linear(hidden_size*4, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        计算两点之间的球面距离（单位：公里）
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371  # 地球平均半径，单位为公里
        return c * r
    @staticmethod
    def bearing(lon1, lat1, lon2, lat2):
        """
        计算两点之间的球面距离（单位：公里）和点2相对于点1的航向角（单位：度）
        航向角以正北为0度，顺时针方向增大。
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式计算距离
        dlon = lon2 - lon1 
        
        # 计算方位角（航向角）
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(y, x)
        bearing = degrees(bearing)  # 将结果从弧度转换为角度
        bearing = (bearing + 360) % 360  # 规范化到0-360度范围
        
        return bearing

    def forward(self, x1, x2, dtime):

        batch_size = x1.shape[0]
        # seq_length = x1.shape[1]
        i = -1
        x1_trimmed = x1[:, i:, :]  # x1_trimmed 现在包含 x1 的后31个点，形状变为 [batch_size, 31, feature_dim]
        #x1_trimmed = x1[:, 1:, :]
        x2_first_point = x2[:, 0:1, :]  # (batch_size, 1, input_size)

        # 计算x1_trimmed最后一个点与x2_first_point的距离
        last_points_x1 = x1[:, -1, :]
        lat1 = last_points_x1[:, 0]
        lon1 = last_points_x1[:, 1]

        lat2 = x2_first_point[:, 0, 0]
        lon2 = x2_first_point[:, 0, 1]

        distances = []
        bearings = []
        distances = torch.tensor([self.haversine(lon1[i].item(), lat1[i].item(), lon2[i].item(), lat2[i].item()) for i in range(batch_size)]).to(x1.device)
        bearings = torch.tensor([self.bearing(lon1[i].item(), lat1[i].item(), lon2[i].item(), lat2[i].item()) for i in range(batch_size)]).to(x1.device)
        bearings = torch.deg2rad(bearings)
        dtime = dtime.view(-1)  # 确保dtime是一个形状为(batch_size,)的张量

        dtime[dtime == 0] = 1  # 将所有 0 值替换为 0.001
         # 更新x2_first_point中的速率
        speeds = distances / dtime * 1000 / 0.514444 # 假设dtime是一个形状为(batch_size,)的张量
        vx = speeds * torch.sin(bearings)
        vy = speeds * torch.cos(bearings)

        # # 更新x2_first_point中的速率
        mask_speeds = (speeds != 0)
        mask_vx = (vx != 0)
        mask_vy = (vy != 0)
        x2_first_point[mask_speeds, 0, 2] = speeds[mask_speeds]

        # 对 vx 和 vy 进行相同的操作
        x2_first_point[mask_vx, 0, 4] = vx[mask_vx]
        x2_first_point[mask_vy, 0, 5] = vy[mask_vy]

        x3 = torch.cat((x1_trimmed, x2_first_point), dim=1)

        x1, x2 ,x3= normalize_coordinates(x1, x2,x3)

        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        x3 = x3.transpose(1, 2)

        # 多尺度卷积
        conv1_out1 = torch.relu(self.conv1(x1))
        conv3_out1 = torch.relu(self.conv3(x1))
        conv5_out1 = torch.relu(self.conv5(x1))
        conv1_out2 = torch.relu(self.conv1(x2))
        conv3_out2 = torch.relu(self.conv3(x2))
        conv5_out2 = torch.relu(self.conv5(x2))
        conv1_out3 = torch.relu(self.conv1(x3))
        conv3_out3 = torch.relu(self.conv3(x3))
        conv5_out3 = torch.relu(self.conv5(x3))

        # 合并不同尺度的特征
        x1 = torch.cat((x1,conv1_out1, conv3_out1, conv5_out1), dim=1).transpose(1, 2)
        x2 = torch.cat((x2,conv1_out2, conv3_out2, conv5_out2), dim=1).transpose(1, 2)
        x3 = torch.cat((x3,conv1_out3, conv3_out3, conv5_out3), dim=1).transpose(1, 2)
        #print(x1.shape)
        h1 = self.feature_extractor(x1,-1)  # (batch_size, 2*hidden_size)
        h2 = self.feature_extractor(x2,0)  # (batch_size, 2*hidden_size)
        h3 = self.feature_extractor(x3,-1)  # (batch_size, 2*hidden_size)
        #print(h2.shape)
        # 计算L1范数距离
        d1 = torch.abs(h1 - h2)  # (batch_size, 2*hidden_size)
        d2 = torch.abs(h1 - h3)  # (batch_size, 2*hidden_size)
        d = torch.cat((d1, d2), dim=-1)

        #print(d.shape)
        # 分类器c
        out = self.fc1(d)
        out = torch.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # (batch_size, 1)
        return out

def normalize_coordinates(x1, x2, x3=None ):
    """
    提取纬度和经度数据，进行墨卡托投影变换，并对坐标进行归一化处理。
    
    参数:
    - x1: 第一组坐标数据 (tensor)，形状为(batch_size, sequence_length, 2)，最后维度代表(纬度, 经度)。
    - x2: 第二组坐标数据 (tensor)，形状与x1相同。
    
    返回:
    - x1_norm: 归一化后的第一组坐标数据。
    - x2_norm: 归一化后的第二组坐标数据。
    """
    # 提取纬度和经度数据
    lat_x1 = x1[:, :, 0]
    lon_x1 = x1[:, :, 1]
    lat_x2 = x2[:, :, 0]
    lon_x2 = x2[:, :, 1]

    lat_x1, lon_x1 = mercator_project(lat_x1, lon_x1)
    lat_x2, lon_x2 = mercator_project(lat_x2, lon_x2)

    # 计算纬度范围
    max_lat_x1, _ = torch.max(lat_x1, dim=1, keepdim=True)
    min_lat_x1, _ = torch.min(lat_x1, dim=1, keepdim=True)
    max_lat_x2, _ = torch.max(lat_x2, dim=1, keepdim=True)
    min_lat_x2, _ = torch.min(lat_x2, dim=1, keepdim=True)

    # 计算总的纬度范围
    max_lat = torch.max(torch.cat([max_lat_x1, max_lat_x2], dim=1), dim=1, keepdim=True)[0]
    min_lat = torch.min(torch.cat([min_lat_x1, min_lat_x2], dim=1), dim=1, keepdim=True)[0]

    # 计算经度范围
    max_lon_x1, _ = torch.max(lon_x1, dim=1, keepdim=True)
    min_lon_x1, _ = torch.min(lon_x1, dim=1, keepdim=True)
    max_lon_x2, _ = torch.max(lon_x2, dim=1, keepdim=True)
    min_lon_x2, _ = torch.min(lon_x2, dim=1, keepdim=True)

    # 计算总的经度范围
    max_lon = torch.max(torch.cat([max_lon_x1, max_lon_x2], dim=1), dim=1, keepdim=True)[0]
    min_lon = torch.min(torch.cat([min_lon_x1, min_lon_x2], dim=1), dim=1, keepdim=True)[0]

    epsilon = 1e-8  # 防止除以零的小常数
    deltalat = max_lat - min_lat + epsilon
    deltalon = max_lon - min_lon + epsilon

    x1[:, :, 0] = (lat_x1 - min_lat) / deltalat
    x2[:, :, 0] = (lat_x2 - min_lat) / deltalat

    x1[:, :, 1] = (lon_x1 - min_lon) / deltalon
    x2[:, :, 1] = (lon_x2 - min_lon) / deltalon

    if x3 is not None:
        lat_x3 = x3[:, :, 0]
        lon_x3 = x3[:, :, 1]
        lon_x3, lat_x3 = mercator_project(lat_x3, lon_x3)
        x3[:, :, 0] = (lat_x3 - min_lat) / deltalat
        x3[:, :, 1] = (lon_x3 - min_lon) / deltalon
        return x1, x2, x3
    else:
        return x1, x2, None

# 注意：normalize_coordinates函数需要被正确实现或导入，这里假设它已经被定义好。
import os.path
# 定义TTCSN模型
class SLITA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SLITA, self).__init__()

        hidden_size = 256
        # 在使用模型前确保设置了CUDA_LAUNCH_BLOCKING
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        # 使用共享权重的BiLSTM
        self.feature_extractor = BiLSTMFeatureExtractor(input_size, hidden_size)
        # 分类器
        self.fc1 = nn.Linear(hidden_size*2, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    def remove_padding(self, x, padding_value=0):
        """去除输入中的填充部分"""
        if isinstance(x, torch.Tensor):
            mask = ~torch.all(x == padding_value, dim=-1)  # 找到非填充部分的掩码
            non_padding_lengths = mask.sum(dim=1)
            outputs = [x[i, :non_padding_lengths[i]] for i in range(x.size(0))]
            return outputs, mask
        else:
            raise ValueError("Unsupported input type. Expected a torch tensor.")
    def forward(self, x1, x2, dtime):
                # 去除填充
        x1, x2 ,_= normalize_coordinates(x1, x2)

        h1 = self.feature_extractor(x1,-1)  # (batch_size, 2*hidden_size)
        h2 = self.feature_extractor(x2,0)  # (batch_size, 2*hidden_size)
        #print(h2.shape)
        # 计算L1范数距离
        d = torch.abs(h1 - h2)  # (batch_size, 2*hidden_size)

        # 分类器
        out = self.fc1(d)
        out = torch.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # (batch_size, 1)
        return out


class BiLSTMFeatureExtractor2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTMFeatureExtractor2, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.bilstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x, idx = -1, idx2 = -1):
        lstm_out, _ = self.bilstm(x)  # lstm_out: (batch_size, seq_len, 2*hidden_size)
        # 取最后一个时间步的输出作为航迹特征
        feature = lstm_out[:, idx, :]  # (batch_size, 2*hidden_size)
        return feature

from math import radians, cos, sin, asin, sqrt,atan2,degrees
class DCSLITA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DCSLITA, self).__init__()

        hidden_size = 256
        # 在使用模型前确保设置了CUDA_LAUNCH_BLOCKING
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        # 使用共享权重的BiLSTM

        self.feature_extractor = BiLSTMFeatureExtractor2(input_size, hidden_size)

        # 分类器
        self.fc1 = nn.Linear(hidden_size*4, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        计算两点之间的球面距离（单位：公里）
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371  # 地球平均半径，单位为公里
        return c * r
    @staticmethod
    def bearing(lon1, lat1, lon2, lat2):
        """
        计算两点之间的球面距离（单位：公里）和点2相对于点1的航向角（单位：度）
        航向角以正北为0度，顺时针方向增大。
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式计算距离
        dlon = lon2 - lon1 

        # 计算方位角（航向角）
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(y, x)
        bearing = degrees(bearing)  # 将结果从弧度转换为角度
        bearing = (bearing + 360) % 360  # 规范化到0-360度范围
        
        return bearing

    def forward(self, x1, x2, dtime):

        # 获取批次大小和序列长度（假设所有序列具有相同的长度）
        batch_size = x1.shape[0]

        # X1 X2是输入的待关联航迹，X1是前段航迹(时间)  X2是后段航迹 
        # dtime 是X1和X2的中断时间长,单位秒  如 X1时间[1,100]  X2时间[200,300] dtime = 99

        #首先对X1截取其后段航迹，这里实际是取最后一个点 x1_trimmed
        i = -1
        x1_trimmed = x1[:, i:, :]  # (batch_size, 1, input_size)

        #对X2取其前段航迹，实际是取第一个点 x2_first_point
        i2 = 1
        x2_first_point = x2[:, 0:i2, :]  # (batch_size, 1, input_size)

        # 计算x1_trimmed与x2_first_point的距离，x2_first_point相对x1_trimmed的方向
        last_points_x1 = x1[:, -1, :]
        lat1 = last_points_x1[:, 0]
        lon1 = last_points_x1[:, 1]

        lat2 = x2_first_point[:, 0, 0]
        lon2 = x2_first_point[:, 0, 1]

        distances = []#距离
        bearings = []#方向
        distances = torch.tensor([self.haversine(lon1[i].item(), lat1[i].item(), lon2[i].item(), lat2[i].item()) for i in range(batch_size)]).to(x1.device)
        bearings = torch.tensor([self.bearing(lon1[i].item(), lat1[i].item(), lon2[i].item(), lat2[i].item()) for i in range(batch_size)]).to(x1.device)
        bearings = torch.deg2rad(bearings)
        dtime = dtime.view(-1)  # 确保dtime是一个形状为(batch_size,)的张量

        dtime[dtime == 0] = 0.001  # 将所有 0 值替换为 0.001
         # 计算中断位移的速率
        speeds = distances / dtime * 1000 / 0.514444 # 假设dtime是一个形状为(batch_size,)的张量
        # 基于速率和方向做矢量分解
        vx = speeds * torch.sin(bearings)
        vy = speeds * torch.cos(bearings)

        #保护一下非0值
        mask_speeds = (speeds != 0)
        mask_vx = (vx != 0)
        mask_vy = (vy != 0)

        # 更新x2_first_point中的速率 VX 和 VY 分别是2，4，5元素
        x2_first_point[mask_speeds, 0, 2] = speeds[mask_speeds]
        x2_first_point[mask_vx, 0, 4] = vx[mask_vx]
        x2_first_point[mask_vy, 0, 5] = vy[mask_vy]

        #将x1_trimmed 和 x2_first_point 拼接起来  这部分是论文中的  时间序列变换
        x3 = torch.cat((x1_trimmed, x2_first_point), dim=1)

        #这里将航迹的经纬度坐标转墨卡托坐标系然后归一
        x1, x2 ,x3= normalize_coordinates(x1, x2,x3)

        #分别对三条航迹提取特征
        h1 = self.feature_extractor(x1,-1)  # (batch_size, 2*hidden_size)
        h2 = self.feature_extractor(x2,0)  # (batch_size, 2*hidden_size)
        h3 = self.feature_extractor(x3,1)

        #计算D2: 中断合理性度量
        d2 = torch.abs(h3 - h1)  # (batch_size, 2*hidden_size)
        # 计算D1 : 航迹形态相似性度量
        d1 = torch.abs(h1 - h2)  # (batch_size, 2*hidden_size)
        #  拼接 实现双路径差异度量
        d = torch.cat((d1, d2), dim=1)

        # 分类器
        out = self.fc1(d)
        out = torch.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # (batch_size, 1)
        return out


class SimplifiedInputEmbedding(nn.Module):
    def __init__(self, D_cont=6, D=64):
        super(SimplifiedInputEmbedding, self).__init__()
        # 简化连续特征映射
        self.W_cont = nn.Linear(D_cont, D)
        
    def forward(self, x):
        # x: [B, seq_len, D_cont]
        cont_emb = self.W_cont(x)  # [B, seq_len, D]
        return cont_emb

# 简化的多头自注意力模块
class SimplifiedMultiHeadSelfAttention(nn.Module):
    def __init__(self, D, h):
        super(SimplifiedMultiHeadSelfAttention, self).__init__()
        self.D = D
        self.h = h
        self.D_h = D // h
        self.W_qkv = nn.Linear(D, 3 * self.D_h * h)
        self.W_msa = nn.Linear(self.D_h * h, D)

    def forward(self, z):
        batch_size, N, _ = z.size()
        qkv = self.W_qkv(z).view(batch_size, N, self.h, 3 * self.D_h).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.D_h ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, N, self.D)
        output = self.W_msa(attn_output)
        return output

# 简化的Transformer编码器块
class SimplifiedTransformerEncoderBlock(nn.Module):
    def __init__(self, D, h):
        super(SimplifiedTransformerEncoderBlock, self).__init__()
        self.msa = SimplifiedMultiHeadSelfAttention(D, h)
        self.ln = nn.LayerNorm(D)

    def forward(self, z):
        z1 = self.ln(z)
        msa_output = self.msa(z1)
        output = z + msa_output
        return output

# 简化的Transformer编码器模块
class SimplifiedTransformerEncoder(nn.Module):
    def __init__(self, D, h, L=2):
        super(SimplifiedTransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([SimplifiedTransformerEncoderBlock(D, h) for _ in range(L)])

    def forward(self, z):
        for block in self.blocks:
            z = block(z)
        return z

class OutputPooling(nn.Module):
    def __init__(self, D):
        super(OutputPooling, self).__init__()
        self.D = D

    def forward(self, z, pooling_strategy='MEAN'):
        z_pooled = None  # 初始化z_pooled为None

        if pooling_strategy == 'MEAN':
            z_pooled = torch.mean(z, dim=1)
        elif pooling_strategy == 'MAX':
            z_pooled = torch.max(z, dim=1)[0]
        elif pooling_strategy == 'CLS':
            z_pooled = z[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}. Choose from ['MEAN', 'MAX', 'CLS']")

        if z_pooled is None:
            raise RuntimeError("Pooling operation did not assign a value to z_pooled. This should not happen.")

        return z_pooled

class TFEITA(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        # 合并输入嵌入和位置编码
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 简化Transformer结构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 动态池化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 相似度计算
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2, dtime):
        # 坐标归一化
        x1, x2, _ = normalize_coordinates(x1, x2)
        
        # 合并特征处理
        def process(x):
            x = self.embed(x)  # [B, L, D]
            x = self.encoder(x)
            x = self.pool(x.transpose(1,2)).squeeze(-1)  # [B, D]
            return x
        
        feat1 = process(x1)
        feat2 = process(x2)
        
        d = torch.abs(feat1 - feat2)  # (batch_size, 2*hidden_size)
        out = torch.relu(d)
        out = self.fc(out)
        # 相似度计算（使用时差加权）

        return torch.sigmoid(out)
    
class DCTFEITA(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        # 合并输入嵌入和位置编码
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 简化Transformer结构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 动态池化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 相似度计算
        self.fc = nn.Linear(hidden_dim*2, 1)
    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        计算两点之间的球面距离（单位：公里）
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371  # 地球平均半径，单位为公里
        return c * r
    @staticmethod
    def bearing(lon1, lat1, lon2, lat2):
        """
        计算两点之间的球面距离（单位：公里）和点2相对于点1的航向角（单位：度）
        航向角以正北为0度，顺时针方向增大。
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式计算距离
        dlon = lon2 - lon1 

        # 计算方位角（航向角）
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(y, x)
        bearing = degrees(bearing)  # 将结果从弧度转换为角度
        bearing = (bearing + 360) % 360  # 规范化到0-360度范围
        
        return bearing

    def forward(self, x1, x2, dtime):

        # 获取批次大小和序列长度（假设所有序列具有相同的长度）
        batch_size = x1.shape[0]
        # 修改 x1_trimmed 以获取 x1 中每个序列的最后一个点
        i = -1
        x1_trimmed = x1[:, i:, :]  # (batch_size, 1, input_size)
        i2 = 1
        x2_first_point = x2[:, 0:i2, :]  # (batch_size, 1, input_size)
        # 计算x1_trimmed最后一个点与x2_first_point的距离
        last_points_x1 = x1[:, -1, :]
        lat1 = last_points_x1[:, 0]
        lon1 = last_points_x1[:, 1]

        lat2 = x2_first_point[:, 0, 0]
        lon2 = x2_first_point[:, 0, 1]

        distances = []
        bearings = []
        distances = torch.tensor([self.haversine(lon1[i].item(), lat1[i].item(), lon2[i].item(), lat2[i].item()) for i in range(batch_size)]).to(x1.device)
        bearings = torch.tensor([self.bearing(lon1[i].item(), lat1[i].item(), lon2[i].item(), lat2[i].item()) for i in range(batch_size)]).to(x1.device)
        bearings = torch.deg2rad(bearings)
        dtime = dtime.view(-1)  # 确保dtime是一个形状为(batch_size,)的张量
        dtime[dtime == 0] = 0.001  # 将所有 0 值替换为 0.001
         # 更新x2_first_point中的速率
        speeds = distances / dtime * 1000 / 0.514444 # 假设dtime是一个形状为(batch_size,)的张量
        vx = speeds * torch.sin(bearings)
        vy = speeds * torch.cos(bearings)
        # # 更新x2_first_point中的速率
        mask_speeds = (speeds != 0)
        mask_vx = (vx != 0)
        mask_vy = (vy != 0)

        x2_first_point[mask_speeds, 0, 2] = speeds[mask_speeds]
        x2_first_point[mask_vx, 0, 4] = vx[mask_vx]
        x2_first_point[mask_vy, 0, 5] = vy[mask_vy]

        x3 = torch.cat((x1_trimmed, x2_first_point), dim=1)
        x1, x2 ,x3= normalize_coordinates(x1, x2, x3)
        
        # 合并特征处理
        def process(x):
            x = self.embed(x)  # [B, L, D]
            x = self.encoder(x)
            x = self.pool(x.transpose(1,2)).squeeze(-1)  # [B, D]
            return x
        
        feat1 = process(x1)
        feat2 = process(x2)
        feat3 = process(x3)
        
        d1 = torch.abs(feat1 - feat2)  # (batch_size, 2*hidden_size)
        d2 = torch.abs(feat1 - feat3)  # (batch_size, 2*hidden_size)
        d = torch.cat((d1, d2), dim=1)
        out = torch.relu(d)
        out = self.fc(out)
        # 相似度计算（使用时差加权）

        return torch.sigmoid(out)