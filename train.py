"""

author: wei guo
24.11.19
content: 实现孪生mlstm模型训练、验证、测试

"""
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import dcfsr_model

#import BaseLine_v1
from tqdm import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point
import contextily as ctx

import os.path
import random



class Batch_CustomDataset:
    def __init__(self, x_paths, y_paths, s_paths, img_paths, max_length=1500):
        self.max_length = max_length
        self.x_data = self._load_data(x_paths, axis=0, dtype=np.float32)  # 加载并转换为 float32
        #self.x_data = self._load_data(x_paths ,axis=0)
        self.y_data = self._load_data(y_paths, axis=0)
        self.s_data = self._load_data(s_paths, axis=0)

    # def _load_data(self, paths,axis):
    #         data_list = []
    #         for path in paths:
    #             data = np.load(path)
    #             if len(data) > self.max_length:
    #                 data = data[:self.max_length]
    #             data_list.append(data)
    #         return np.concatenate(data_list, axis=axis)
    def _load_data(self, paths, axis=0, dtype=None):
        """
        加载数据的辅助方法。
        :param paths: 数据文件路径列表
        :param axis: 合并数据的轴，默认为 0
        :param dtype: 数据类型（可选）
        :return: 加载后的 NumPy 数组
        """
        data_list = []
        for path in paths:
            try:
                # 加载数据
                data = np.load(path)
                
                # 限制数据长度
                if len(data) > self.max_length:
                    data = data[:self.max_length]
                
                # 如果指定了 dtype，则进行类型转换
                if dtype is not None:
                    data = data.astype(dtype)
                
                data_list.append(data)
            except FileNotFoundError:
                print(f"File not found: {path}")
                raise
            except ValueError as e:
                print(f"Error loading file {path}: {e}")
                raise
        
        # 合并所有数据
        return np.concatenate(data_list, axis=axis)
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        features = torch.tensor(self.x_data[idx], dtype=torch.float32)
        # # 从路径加载图像
        # img_path = self.img_paths[idx]

        # img1 = torch.tensor(cv2.imread(img_path[0]), dtype=torch.float32)
        # img1 = img_trans(img1)
        # img2 = torch.tensor(cv2.imread(img_path[1]), dtype=torch.float32)
        # img2 = img_trans(img2)

        labels = torch.tensor(self.y_data[idx], dtype=torch.float32)
        return features, labels

class CustomSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
def split_dataset_by_scenarios(dataset, train_size=0.7, val_size=0.1, test_size=0.2, random_state=None):
    unique_scenarios = np.unique(dataset.s_data)
    # 划分场景
    train_scenarios, temp_scenarios = train_test_split(unique_scenarios, train_size=train_size, random_state=random_state)
    val_scenarios, test_scenarios = train_test_split(temp_scenarios, train_size=val_size/(val_size + test_size), random_state=random_state)

    indices = np.arange(len(dataset.s_data))
    train_indices = np.isin(dataset.s_data, train_scenarios)
    val_indices = np.isin(dataset.s_data, val_scenarios)
    test_indices = np.isin(dataset.s_data, test_scenarios)

    train_dataset = CustomSubset(dataset, indices[train_indices])
    val_dataset = CustomSubset(dataset, indices[val_indices])
    test_dataset = CustomSubset(dataset, indices[test_indices])

    return train_dataset, val_dataset, test_dataset
def create_balanced_sampler1(labels, positive_ratio):
    """
    创建一个加权随机采样器，允许设置正样本与负样本的比例。
    
    :param labels: 标签数组，用于计算各类别的样本数量。
    :param positive_ratio: 正样本相对于总样本的比例。
    :return: 一个WeightedRandomSampler对象。
    """
    # 获取类别列表和各类别的样本数量
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    n_classes = len(class_sample_count)

    # 假设二分类问题，其中标签0为负样本，1为正样本
    if n_classes != 2:
        raise ValueError("此函数仅适用于二分类问题")

    # 计算期望的正样本数量
    total_samples = len(labels)
    desired_positive_samples = int(total_samples * positive_ratio)
    desired_negative_samples = total_samples - desired_positive_samples

    # 根据期望的样本数量调整权重
    weight_per_class = [desired_negative_samples / class_sample_count[0], desired_positive_samples / class_sample_count[1]]
    samples_weight = np.array([weight_per_class[t] for t in labels])

    # 转换为PyTorch张量并创建WeightedRandomSampler
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples=total_samples, replacement=True)
    
    return sampler
def split_dataset_by_ratio(dataset, train_size=0.7, val_size=0.1, test_size=0.2, random_state=None):
    # 确保比例之和为 1
    #assert train_size + val_size + test_size == 1, "The sum of train_size, val_size and test_size must be 1."

    # 生成数据集的索引
    indices = np.arange(len(dataset))

    # 第一次划分：划分训练集和临时集（包含验证集和测试集）
    train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=random_state)

    # 计算验证集在临时集中的比例
    val_ratio_in_temp = val_size / (val_size + test_size)

    # 第二次划分：从临时集中划分验证集和测试集
    val_indices, test_indices = train_test_split(temp_indices, train_size=val_ratio_in_temp, random_state=random_state)

    # 创建自定义子集
    train_dataset = CustomSubset(dataset, train_indices)
    val_dataset = CustomSubset(dataset, val_indices)
    test_dataset = CustomSubset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


from math import radians, cos, sin, asin, sqrt
def haversine(lat1, lon1, lat2, lon2):
    """
    计算两点间的球面距离（单位：公里）
    所有参数都应该是形状相同的张量
    """
    R = 6371.0  # 地球半径，单位为公里

    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)

    a = torch.sin(dlat / 2)**2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = R * c * 1000
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从点 (lat1, lon1) 到点 (lat2, lon2) 的方位角（单位：度）
    """
    # 将纬度和经度从度转换为弧度
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    # 计算差值
    dlon_rad = lon2_rad - lon1_rad

    # 计算 y 和 x
    y = torch.sin(dlon_rad) * torch.cos(lat2_rad)
    x = torch.cos(lat1_rad) * torch.sin(lat2_rad) - \
        torch.sin(lat1_rad) * torch.cos(lat2_rad) * torch.cos(dlon_rad)

    # 计算方位角（弧度）
    bearing_rad = torch.atan2(y, x)

    # 转换为角度，并确保范围在 [0, 360]
    bearing_deg = torch.rad2deg(bearing_rad)
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


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

def selectData(dataset, cols = [0,1,2,3]):
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    x1,x2 = dataset[:,0,:,:].cuda(),dataset[:,1,:,:].cuda()
    
    x1in = x1[:,:,cols]
    x2in = x2[:,:,cols]
    
    if isNormalTime == True:
        
        x2_time_max = torch.max(x2in[:, :, 3], dim=1, keepdim=True)[0]
        x1_time_min = torch.min(x1in[:, :, 3], dim=1, keepdim=True)[0]
        # 避免除以0
        epsilon = 1e-8  # 或者一个更小的值，取决于你的数据范围
        x2_time_max[x2_time_max == 0] = epsilon
        range_time = x2_time_max - x1_time_min
        x2_time_min = torch.min(x2in[:, :, 3], dim=1, keepdim=True)[0]

        x1_time_max = torch.max(x1in[:, :, 3], dim=1, keepdim=True)[0]
        dtmin = x1_time_max - x1_time_min
        
        #print(x1in[:, :, 3])
        #time1 = (x1in[:, :, 3] - x1_time_min) / range_time * 100
        x1in[:, :, 3] = (x1in[:, :, 3] - x1_time_min) / range_time * 100
        x2in[:, :, 3] = (x2in[:, :, 3] - x1_time_min) / range_time * 100

    x1_time_max = torch.max(x1[:, :, [3]], dim=1, keepdim=True)[0]
    x2_time_min = torch.min(x2[:, :, [3]], dim=1, keepdim=True)[0]    
    dtime = x2_time_min - x1_time_max

    return x1in,x2in,dtime

from torch.utils.data import DataLoader, WeightedRandomSampler
def create_balanced_sampler(labels):
    # 计算每个类别的权重
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler
# x_savep = r'D:\断续航迹关联（全中断）\数据集v5_1224\600\x.npy'
# y_savep = r'D:\断续航迹关联（全中断）\数据集v5_1224\600\y.npy'
# img_savep = r'D:\断续航迹关联（全中断）\数据集v5_1224\600\imgs.npy'
# dataset = CustomDataset(x_savep, y_savep,img_savep)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
import os
def remove_padding(inputs, padding_value=0):
    """去除输入中的填充部分"""
    if isinstance(inputs, np.ndarray):
        # 假设填充在序列末尾，找到非填充部分的长度
        non_padding_lengths = np.argmax(inputs == padding_value, axis=1)
        # 如果没有找到填充，则使用整个序列长度
        for i in range(len(non_padding_lengths)):
            if non_padding_lengths[i] == 0 and (inputs[i, 0] != padding_value).all():
                non_padding_lengths[i] = inputs.shape[1]
        # 移除填充
        outputs = [inputs[i, :non_padding_lengths[i]] for i in range(inputs.shape[0])]
        return outputs
    else:
        raise ValueError("Unsupported input type. Expected a numpy array.")

#//////////////////////////////////////////////////////////////////////////

#dir = r'D:\AISData\us\AIS_2023_12_31\traindata'
dir = r'D:\AISData\us\big\train'

model_save_path = r'D:\work\pypro\model_new\TEST.pt'
learning_rate = 0.0001
epochs= 40
epoch_list = [10,20,30,40,80,100,150,170,200,250,270,300]

select_trandata = [0,1,2,3,5,6]
isNormalTime = True

# 初始化模型
model = dcfsr_model.DCTFEITA(len(select_trandata),256).cuda()

xname = 'inter_x_fcy.npy'
yname = 'inter_y_fcy.npy'
sname = 'inter_s_fcy.npy'

xpaths,ypaths, spaths,imgpaths = [],[],[],[]
xpaths.append(os.path.join(dir,xname))
ypaths.append(os.path.join(dir,yname))
spaths.append(os.path.join(dir,sname))

torch.manual_seed(42)
dataset = Batch_CustomDataset(xpaths,ypaths,spaths,imgpaths,max_length=50000 )
train_dataset, val_dataset, test_dataset = split_dataset_by_ratio(dataset, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42)
# # 设置随机种子
batch_size = 512
#torch.manual_seed(11)
# 创建DataLoader

samplep = 0.3
train_sampler = create_balanced_sampler1(train_dataset.dataset.y_data[train_dataset.indices], samplep)
val_sampler = create_balanced_sampler1(val_dataset.dataset.y_data[val_dataset.indices], samplep)
test_sampler = create_balanced_sampler1(test_dataset.dataset.y_data[test_dataset.indices], samplep)


# 使用 DataLoader 并传入 sampler
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

print('train_dataset:',len(train_dataset))
print('val_dataset:',len(val_dataset))
print('test_dataset:',len(test_dataset))


# 假设模型已经在GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 定义缩放变换
resize_transform = transforms.Resize((224, 224))

train_losses = []
val_losses = []
best_loss = 9999999
for epoch in range(epochs):

    model.train()
    train_loss = 0
    for i, (inputs,targets) in enumerate(train_dataloader):
        #print(inputs.shape)
        #print(inputs)
        x1in, x2in,dtime = selectData(inputs, select_trandata)

        # 将输入数据移动到相同的设备
        x1in = x1in.to(device)
        x2in = x2in.to(device)
        dtime = dtime.to(device)
        targets = targets.to(device)

        outputs = model(x1in, x2in, dtime).squeeze()

        loss = criterion(outputs, targets.to(outputs.device))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = torch.flatten(outputs)
        outputs.unsqueeze(1)
        
        train_loss += loss.item()

        #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    train_loss /=len(train_dataloader)
    train_losses.append(train_loss)

    if train_loss<best_loss:
        best_loss = train_loss
        torch.save(model ,model_save_path)
        print(epoch , '- save best model')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features,labels in test_dataloader:

            x1in1, x2in1,dtime = selectData(features, select_trandata)

            outputs = model(x1in1,x2in1, dtime)

            outputs = torch.flatten(outputs)
            outputs.unsqueeze(1)

            loss = criterion(outputs, labels.cuda())
            val_loss += loss.item()

            # 应用阈值函数，大于0.5取1，否则取0
            predicted = (outputs.cpu() > 0.5).float()
            total += labels.cpu().size(0)

            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    if (epoch+1)%1 == 0:
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f},'
              f'Val Acc:{accuracy:.6f}')

    
    if epoch+1 in epoch_list:
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        wrong_samples = []  # 用于存储错误样本的信息
        with torch.no_grad():
            for features, labelst in test_dataloader:

                x1in1, x2in1,dtime = selectData(features, select_trandata)
                # 使用模型进行预测
                outputs = model(x1in1,x2in1, dtime)

                outputs = torch.flatten(outputs)
                outputs.unsqueeze(1)

                # 应用阈值函数，大于0.5取1，否则取0
                predicted = (outputs.cpu() > 0.5).float()
                total += labelst.cpu().size(0)
                predicted = torch.flatten(predicted)
                # print(predicted, labels)
                correct += (predicted == labelst).sum().item()

                # 检查哪些样本是错误分类的
                mismatches = (predicted != labelst).nonzero(as_tuple=True)[0]
                for idx in mismatches:
                    wrong_samples.append({
                        'features': features[idx].cpu().numpy(),
                        'label': labelst[idx].cpu().item(),
                        'predicted': predicted[idx].item()
                    })

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labelst.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

        # 初始化标签计数器
        label_counts = {0: 0, 1: 0}

        # 提取所有错误样本的航迹数据并统计标签数量
        for sample_idx, sample in enumerate(wrong_samples):
            label = sample['label']
            
            # 更新标签计数器
            if label in label_counts:
                label_counts[label] += 1
            else:
                print(f"Warning: Unexpected label {label} found in sample {sample_idx}.")
                
        # 打印标签为 0 和 1 的总数
        #print(all_labels)
        num_labels = len(all_labels)
        print(type(all_labels))
        #num_ones = np.sum(all_labels == 1.0)
        num_ones = len([x for x in all_labels if x == 1.0])
        
        print(f"Total number of samples: {num_labels}")
        print(f"Total number of samples with 1: {num_ones}")
        print(f"Total number of wrong_samples with label 0: {label_counts[0]}")
        print(f"Total number of wrong_samples with label 1: {label_counts[1]}")

# 绘制训练和验证损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(model_save_path.split('.')[0]+'.jpg')
plt.show()


