from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image
import os.path
import clip
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
#import BaseLine_v1
from tqdm import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# 定义自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, x_path, y_path,img_path):
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)
        self.img_data = np.load(img_path)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        features = torch.tensor(self.x_data[idx], dtype=torch.float32)
        img = self.img_data[idx]
        labels = torch.tensor(self.y_data[idx], dtype=torch.float32)
        return features, img,labels

# 定义自定义Dataset类
from random import sample
from random import randint
class Batch_CustomDataset:
    def __init__(self, x_paths, y_paths, s_paths, img_paths, m_paths, max_length=6000):
        self.max_length = max_length
        
        # 1. 加载 s_data 并提取场景名称
        self.s_data, scene_names = self._load_s_data_with_scenes(s_paths)
        
        # 2. 统计不同的场景名称
        unique_scenes = list(set(scene_names))
        
        # 3. 随机选择 100 个场景（或所有场景，如果不足 100 个）
        if len(unique_scenes) > 100:
            selected_scenes = sample(unique_scenes, 100)
        else:
            selected_scenes = unique_scenes
        
        # 4. 筛选与选定场景相关的索引
        selected_indices = [i for i, scene in enumerate(scene_names) if scene in selected_scenes]
        
        # 根据选定的索引加载其他数据
        self.x_data = self._load_selected_data(x_paths, selected_indices, axis=0, dtype=np.float32)
        self.y_data = self._load_selected_data(y_paths, selected_indices, axis=0)
        self.m_data = self._load_selected_data(m_paths, selected_indices, axis=0)

        # 筛选 s_data
        self.s_data = self.s_data[selected_indices]

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

    def _load_s_data_with_scenes(self, paths):
        """
        加载 s_data 并提取场景名称。
        :param paths: 数据文件路径列表
        :return: 加载后的 s_data 和对应的场景名称列表
        """
        data_list = []
        scene_names = []
        for path in paths:
            try:
                # 加载数据
                data = np.load(path)
                
                # 限制数据长度
                if len(data) > self.max_length:
                    data = data[:self.max_length]
                
                # 假设每个文件的数据直接表示场景名称
                scene_names.extend(data.tolist())  # 将场景名称添加到列表
                
                data_list.append(data)
            except FileNotFoundError:
                print(f"File not found: {path}")
                raise
            except ValueError as e:
                print(f"Error loading file {path}: {e}")
                raise
        
        # 合并所有数据
        return np.concatenate(data_list, axis=0), scene_names

    def _load_selected_data(self, paths, selected_indices, axis=0, dtype=None):
        """
        根据选定的索引加载数据。
        :param paths: 数据文件路径列表
        :param selected_indices: 选定的索引列表
        :param axis: 合并数据的轴，默认为 0
        :param dtype: 数据类型（可选）
        :return: 加载后的 NumPy 数组
        """
        all_data = self._load_data(paths, axis=axis, dtype=dtype)
        return all_data[selected_indices]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # 随机选择一个索引
        random_idx = randint(0, len(self.x_data) - 1)
        
        # 获取随机索引对应的特征、标签、MMSI 和场景
        features = torch.tensor(self.x_data[random_idx], dtype=torch.float32)
        labels = torch.tensor(self.y_data[random_idx], dtype=torch.float32)
        mmsis = self.m_data[random_idx]  # Python list 或者 单个字符串值
        scenes = self.s_data[random_idx]

        # 返回随机抽取的样本
        return features, labels, mmsis, scenes
def get_device():
    if torch.cuda.is_available():
        # 获取当前可用的GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        # 如果有多个GPU，可以选择第一个可用的GPU
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU.")
    
    return device

device = torch.device(2)
# 定义缩放变换
resize_transform = transforms.Resize((224, 224))

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

def plot_single_trajectory(trajectory_tensor, sample_index):
    """
    绘制单条轨迹
    :param features: 包含所有特征的数据
    :param idx: 需要绘制的轨迹索引
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for track in range(2):  # 遍历两条航迹
        lat_lon_data = trajectory_tensor[sample_index, track, :, :2]  # 提取纬度和经度 [64, 2]
        lat_lon_np = lat_lon_data.cpu().numpy()  # 如果在CPU上运行则不需要.cpu()
        time_data = trajectory_tensor[sample_index, track, :, 3]  # 提取时间数据 [64]
        
        # 绘制整条轨迹
        ax.plot(lat_lon_np[:, 1], lat_lon_np[:, 0], marker='o', label=f'Track {track+1}', linestyle='-')
        
        # 获取并打印起始时间和结束时间
        start_time = time_data[0].item()  # 获取第一条记录的时间
        end_time = time_data[-1].item()  # 获取最后一条记录的时间
        print(f"Track {track + 1}: Start Time = {start_time}, End Time = {end_time}")
        
        # 标记起点和终点
        ax.scatter(lat_lon_np[0, 1], lat_lon_np[0, 0], color='red', zorder=5, label='Start Point' if track == 0 else None)  # 起点
        ax.scatter(lat_lon_np[-1, 1], lat_lon_np[-1, 0], color='green', zorder=5, label='End Point' if track == 0 else None)  # 终点
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title(f'Sample {sample_index} Trajectories Plot')
    plt.legend()
    plt.show()

def save_single_trajectory(trajectory_tensor, sample_index, save_path):
    """
    绘制单条轨迹并保存图像。
    
    :param trajectory_tensor: 包含所有特征的数据
    :param sample_index: 需要绘制的轨迹索引
    :param save_path: 图像保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for track in range(2):  # 遍历两条航迹
        lat_lon_data = trajectory_tensor[sample_index, track, :, :2]  # 提取纬度和经度 [64, 2]
        lat_lon_np = lat_lon_data.cpu().numpy() if hasattr(lat_lon_data, 'cpu') else lat_lon_data.numpy()  # 确保数据在CPU上
        time_data = trajectory_tensor[sample_index, track, :, 3]  # 提取时间数据 [64]

        # 绘制整条轨迹
        ax.plot(lat_lon_np[:, 1], lat_lon_np[:, 0], marker='o', label=f'Track {track+1}', linestyle='-')

        # 获取并打印起始时间和结束时间
        start_time = time_data[0].item()  # 获取第一条记录的时间
        end_time = time_data[-1].item()  # 获取最后一条记录的时间
        #print(f"Track {track + 1}: Start Time = {start_time}, End Time = {end_time}")

        # 标记起点和终点
        ax.scatter(lat_lon_np[0, 1], lat_lon_np[0, 0], color='red', zorder=5, label='Start Point' if track == 0 else None)  # 起点
        ax.scatter(lat_lon_np[-1, 1], lat_lon_np[-1, 0], color='green', zorder=5, label='End Point' if track == 0 else None)  # 终点

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title(f'Sample {sample_index} Trajectories Plot')
    plt.legend()

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图形，防止内存泄漏

def gethr1(sceen_interval, mmsi_pairs, y_outs, y_true, y_pred):
    # 记录每条航迹得分的最大值
    mmsi_max_scores = {}  # 用于存储每个 MMSI 的最大得分

    HR1LIST = []

    for i, pair in enumerate(mmsi_pairs):
        if y_true[i] == 0:  # 只处理真实标签为 0 的情况
            mmsi1, mmsi2 = pair.split(":")  # 分割航迹对的 MMSI
            mmsi1 = mmsi1 + sceen_interval[i]
            mmsi2 = mmsi2 + sceen_interval[i]
            score = y_outs[i]  # 获取当前航迹对的得分
            
            # 更新 mmsi1 的最大得分
            if mmsi1 in mmsi_max_scores:
                mmsi_max_scores[mmsi1] = max(mmsi_max_scores[mmsi1], score)
            else:
                mmsi_max_scores[mmsi1] = score
            
            # 更新 mmsi2 的最大得分
            if mmsi2 in mmsi_max_scores:
                mmsi_max_scores[mmsi2] = max(mmsi_max_scores[mmsi2], score)
            else:
                mmsi_max_scores[mmsi2] = score

    count_exceeding = 0  # 记录 y_outs 大于 mmsi_max_scores 中数值的数量

    for i, pair in enumerate(mmsi_pairs):
        if (y_true[i] == 1) and (y_pred[i] == 1):  # 只处理真实标签为 1 的情况
            mmsi1, mmsi2 = pair.split(":")  # 分割航迹对的 MMSI
            mmsi1 = mmsi1 + sceen_interval[i]
            mmsi2 = mmsi2 + sceen_interval[i]
            score = y_outs[i]  # 获取当前航迹对的得分
            
            # 检查 mmsi_max_scores 中是否存在对应的 MMSI
            if mmsi1 in mmsi_max_scores and mmsi2 in mmsi_max_scores:
                # 如果 y_outs 大于任意一条航迹的最大得分，则计数加 1
                if score > mmsi_max_scores[mmsi1] or score > mmsi_max_scores[mmsi2]:
                    count_exceeding += 1
                    HR1LIST.append(1)
                else:
                    HR1LIST.append(0)
            else:
                #print('///////////////WWWWWWWWWWWWWWWWWWWWWWWWWWRONG!!!!!!!!!!!!')
                HR1LIST.append(0)
                count_exceeding += 1
        else:
            HR1LIST.append(0)
            count_exceeding += 1
    
    #count_labels_1 += (labels == 1).sum().item()
    #count_y_true_1 = y_true.count(1)
    #hr1 = count_exceeding / count_y_true_1

    return HR1LIST
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
def dowork(iteration, dataset,result_dir,model_save_path):
    """
    单次运行的主函数，添加 iteration 参数用于标识当前运行次数。
    """
    results = []  # 在循环外部初始化一个空列表来存储结果

    batch_size = 64 * 8
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #model_save_path = r'D:\work\pypro\model_new\DCTA_C_US_012356.pt'
    device = get_device()
    model = torch.load(model_save_path, map_location=device).to(device)

    model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    d_time = []
    select_trandata = [0, 1, 2, 3, 5, 6]
    isNormalTime = True

    mmsi_pairs = []
    y_outs = []
    scenes_pairs = []

    with torch.no_grad():
        for features, labels, mmsis, scenes in test_dataloader:
            wrong_indices = []

            x1, x2 = features[:, 0, :, :].to(device), features[:, 1, :, :].to(device)
            x1in = x1[:, :, select_trandata]
            x2in = x2[:, :, select_trandata]

            #时间处理
            if isNormalTime:
                x2_time_max = torch.max(x2in[:, :, 3], dim=1, keepdim=True)[0]
                x1_time_min = torch.min(x1in[:, :, 3], dim=1, keepdim=True)[0]
                x2_time_min = torch.min(x2in[:, :, 3], dim=1, keepdim=True)[0]
                epsilon = 1e-8
                x2_time_max[x2_time_max == 0] = epsilon
                range_time = x2_time_max - x1_time_min

                x1in[:, :, 3] = (x1in[:, :, 3] - x1_time_min) / range_time * 100
                x2in[:, :, 3] = (x2in[:, :, 3] - x1_time_min) / range_time * 100

            x1_time_max = torch.max(x1[:, :, [3]], dim=1, keepdim=True)[0]
            x2_time_min = torch.min(x2[:, :, [3]], dim=1, keepdim=True)[0]
            dtime = x2_time_min - x1_time_max


            outputs = model(x1in, x2in, dtime)
            outputs = torch.flatten(outputs)
            outputs.unsqueeze(1)

            d_time.append(dtime)

            mmsi_pairs.extend(mmsis)
            scenes_pairs.extend(scenes)
            y_outs.extend(outputs.cpu())
            predicted = (outputs.cpu() > 0.5).float()
            total += labels.cpu().size(0)

            for idx, (p, l) in enumerate(zip(predicted, labels)):
                if p != l:
                    wrong_indices.append(idx)

            predicted = torch.flatten(predicted)
            correct += (predicted == labels).sum().item()
            y_pred.extend(predicted)
            y_true.extend(labels.cpu().numpy())

    hr1list = gethr1(scenes_pairs, mmsi_pairs, y_outs, y_true, y_pred)
    hr1 = hr1list.count(1) / y_true.count(1)

    intervals = [(0, 1200), (1200, 2400), (2400, 3600), (3600, 4800), (4800, 7200)]
    results = []
    columns = {}

    d_time_np = torch.cat(d_time).cpu().numpy()

    for idx, (lower, upper) in enumerate(intervals):
        mask_interval = (d_time_np >= lower) & (d_time_np < upper)
        indices = np.where(mask_interval)[0]

        yt_interval = [y_true[i] for i in indices]
        yp_interval = [y_pred[i] for i in indices]
        mmsis_interval = [mmsi_pairs[i] for i in indices]
        youts_interval = [y_outs[i] for i in indices]
        sceen_interval = [scenes_pairs[i] for i in indices]

        if len(yt_interval) == 0:
            results.append([lower, upper, None, None, None])
            continue

        accuracy_interval = accuracy_score(yt_interval, yp_interval)
        conf_matrix_interval = confusion_matrix(yt_interval, yp_interval)
        tn, fp, fn, tp = conf_matrix_interval.ravel()
        fpr = fp / (fp + tn)

        formatted_accuracy = "{:.4f}".format(accuracy_interval)
        formatted_precision_score = "{:.4f}".format(precision_score(yt_interval, yp_interval))
        formatted_recall_score = "{:.4f}".format(recall_score(yt_interval, yp_interval))
        formatted_f1_score = "{:.4f}".format(f1_score(yt_interval, yp_interval))
        formatted_fpr_score = "{:.4f}".format(fpr)

        hr1LIST_I = gethr1(sceen_interval, mmsis_interval, youts_interval, yt_interval, yp_interval)
        hr1_I = hr1LIST_I.count(1) / yt_interval.count(1)
        formatted_hr1_score = "{:.4f}".format(hr1_I)

        results.append({
            'Accuracy': formatted_accuracy,
            'Precision': formatted_precision_score,
            'Recall': formatted_recall_score,
            'F1-Score': formatted_f1_score,
            'FPR': formatted_fpr_score,
            'HR1': formatted_hr1_score
        })

        columns[f'Accuracy_{idx}'] = formatted_accuracy
        columns[f'Precision_{idx}'] = formatted_precision_score
        columns[f'Recall_{idx}'] = formatted_recall_score
        columns[f'F1-Score_{idx}'] = formatted_f1_score
        columns[f'FPR_{idx}'] = formatted_fpr_score
        columns[f'HR1_{idx}'] = formatted_hr1_score

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn)

    overall_accuracy = "{:.4f}".format(accuracy_score(y_true, y_pred))
    overall_precision = "{:.4f}".format(precision_score(y_true, y_pred))
    overall_recall = "{:.4f}".format(recall_score(y_true, y_pred))
    overall_f1 = "{:.4f}".format(f1_score(y_true, y_pred))
    overall_fpr = "{:.4f}".format(fpr)
    overall_hr1 = "{:.4f}".format(hr1)

    results.append({
        'Accuracy': overall_accuracy,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1-Score': overall_f1,
        'FPR': overall_fpr,
        'HR1': overall_hr1
    })

    columns['Overall_Accuracy'] = overall_accuracy
    columns['Overall_Precision'] = overall_precision
    columns['Overall_Recall'] = overall_recall
    columns['Overall_F1-Score'] = overall_f1
    columns['Overall_FPR'] = overall_fpr
    columns['Overall_HR1'] = overall_hr1

    df_results = pd.DataFrame(results)
    output_file_path = os.path.join(result_dir, f'model_evaluation_results_{iteration}.csv')
    df_results.to_csv(output_file_path, index=False)

    for key, value in columns.items():
        columns[key] = [value]

    ordered_columns = []
    for i in range(len(columns) // 6):  # 每次循环新增 6 列
        if i != 5:
            ordered_columns.extend([f'Accuracy_{i}', f'Precision_{i}', f'Recall_{i}', f'F1-Score_{i}', f'FPR_{i}', f'HR1_{i}'])
        else:
            ordered_columns.extend(['Overall_Accuracy', 'Overall_Precision', 'Overall_Recall', 'Overall_F1-Score', 'Overall_FPR', 'Overall_HR1'])

    df_results1 = pd.DataFrame(columns)[ordered_columns]
    output_file_path1 = os.path.join(result_dir, f'model_evaluation_results1_{iteration}.csv')
    df_results1.to_csv(output_file_path1, index=False)


result_dirs = []
models = []
# 定义全局变量
dir = r'D:\AISData\dma\big\test'
result_dirs.append(r'D:\AISData\results_dcta_dma')
result_dirs.append(r'D:\AISData\results_ttcsn_dma')
result_dirs.append(r'D:\AISData\results_mcil_dma')
result_dirs.append(r'D:\AISData\results_TSACTFER_dma')
result_dirs.append(r'D:\AISData\results_TSACTFER2_dma')
result_dirs.append(r'D:\AISData\results_dcmcil_dma')

# models.append(r'D:\work\pypro\model_new\DCTA_DMA_012356.pt')
# models.append(r'D:\work\pypro\model_new\TTCSN_DMA_012356.pt')
# models.append(r'D:\work\pypro\model_new\MCIL_DMA_012356.pt')
# models.append(r'D:\work\pypro\model_new\TSACTFER_DMA_012356.pt')
# models.append(r'D:\work\pypro\model_new\TSACTFER2_DMA_012356.pt')
# models.append(r'D:\work\pypro\model_new\DCMCIL_DMA_012356.pt')

# result_dirs.append(r'D:\AISData\results_dcta_us')
# result_dirs.append(r'D:\AISData\results_ttcsn_us')
# result_dirs.append(r'D:\AISData\results_mcil_us')
# result_dirs.append(r'D:\AISData\results_TSACTFER_us')
# result_dirs.append(r'D:\AISData\results_TSACTFER2_us')
# result_dirs.append(r'D:\AISData\results_dcmcil_us')

models.append(r'D:\work\pypro\model_new\DCTA_US_012356.pt')
models.append(r'D:\work\pypro\model_new\TTCSN_US_012356.pt')
models.append(r'D:\work\pypro\model_new\MCIL_US_012356.pt')
models.append(r'D:\work\pypro\model_new\TSACTFER_US_012356.pt')
models.append(r'D:\work\pypro\model_new\TSACTFER2_US_012356.pt')
models.append(r'D:\work\pypro\model_new\DCMCIL_US_012356.pt')


xname = 'inter_x_fcy.npy'
yname = 'inter_y_fcy.npy'
sname = 'inter_s_fcy.npy'
mname = 'inter_m_fcy.npy'
imgname = 'inter_imgs_fcy.npy'


# 循环运行 50 次
for i in range(50):

    xpaths, ypaths, spaths, imgpaths, mpaths = [], [], [], [], []
    xpaths.append(os.path.join(dir, xname))
    ypaths.append(os.path.join(dir, yname))
    spaths.append(os.path.join(dir, sname))
    imgpaths.append(os.path.join(dir, imgname))
    mpaths.append(os.path.join(dir, mname))

    dataset = Batch_CustomDataset(xpaths, ypaths, spaths, imgpaths, mpaths, max_length=10000000)

    print(f"Running iteration {i + 1}...")
    # 内层循环：次切换不同的 result_dirs
    for j, result_dir in enumerate(result_dirs):
        print(f"  Sub-iteration {j + 1}: Using result directory {result_dir}")
        dowork(i + 1, dataset, result_dir,models[j])