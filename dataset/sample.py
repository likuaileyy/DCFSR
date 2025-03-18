
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import  pandas as pd
import numpy as np
import torch
from tqdm import *
#from modelscope import AutoModelForCausalLM, AutoTokenizer
import cv2
from scipy.interpolate import CubicSpline
warnings.filterwarnings('ignore')
import math
import gc


# 定义字段名称变量
mmsi_field = 'MMSI'
base_date_time_field = 'BaseDateTime'
lat_field = 'LAT'
lon_field = 'LON'
sog_field = 'SOG'
cog_field = 'COG'
#heading_field = 'Heading'
time_field = 'BaseDateTime'  # 假设时间字段名为'time'，根据实际情况调整
vx_field = 'vx'
vy_field = 'vy'
vx1_field = 'vx1'
vy1_field = 'vy1'
v1_field = 'v1'

psample = 1

def haversine(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # 地球平均半径，单位为公里
    r = 6371
    return c * r *1000
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算两点之间的航向角（单位：度）。
    """
    # 将经纬度转换为弧度
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    lambda1, lambda2 = np.radians(lon1), np.radians(lon2)

    # 计算差值
    delta_lambda = lambda2 - lambda1

    # 计算 y 和 x 分量
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    # 计算初始航向角（弧度）
    initial_bearing = np.arctan2(y, x)

    # 转换为角度，并确保范围在 [0, 360)
    #bearing = np.degrees(initial_bearing)
    #bearing = (bearing + 360) % 360

    return initial_bearing


import threading
    # 定义一个锁对象
lock = threading.Lock()

X, y, S, M, imgs, csv_names = [], [], [], [], [], []

def process_single_file(name, dir, img_p, max_trace_len):
    path = os.path.join(dir, name)
    tx, ty, tdf, timgs, ts, tm = csv2npy_withInter(path, img_p, max_trace_len, max_num=1000, isTimeStd=True)
    if tx is not None:
        return (tx, ty, ts, tm, timgs, name)
    return None

def update_shared_lists(result):
    if result is not None:
        tx, ty, ts, tm, timgs, name = result
        with lock:  # 使用锁保护对共享资源的访问
            X.extend(tx)
            y.extend(ty)
            S.extend(ts)
            M.extend(tm)
            imgs.extend(timgs)
            csv_names.append(name)

from threading import Lock
# 初始化锁
lock_npy = Lock()

def update_shared_lists1(result, x_savep, y_savep, s_savep, m_savep):
    """
    直接保存处理结果到指定的 .npy 文件中。
    """
    if result is None:
        return

    tx, ty, ts, tm, timgs, name = result

    with lock_npy:  # 使用锁保护文件写入操作
        try:
            # 将当前结果转换为 NumPy 数组
            tx_array = np.array(tx) if tx else np.array([])
            ty_array = np.array(ty) if ty else np.array([])
            ts_array = np.array(ts) if ts else np.array([])
            tm_array = np.array(tm) if tm else np.array([])

            # 加载已有的数据（如果文件存在）
            existing_x = np.load(x_savep, allow_pickle=True) if os.path.exists(x_savep) else np.array([])
            existing_y = np.load(y_savep, allow_pickle=True) if os.path.exists(y_savep) else np.array([])
            existing_s = np.load(s_savep, allow_pickle=True) if os.path.exists(s_savep) else np.array([])
            existing_m = np.load(m_savep, allow_pickle=True) if os.path.exists(m_savep) else np.array([])

            # 追加新数据
            updated_x = np.concatenate([existing_x, tx_array]) if existing_x.size > 0 else tx_array
            updated_y = np.concatenate([existing_y, ty_array]) if existing_y.size > 0 else ty_array
            updated_s = np.concatenate([existing_s, ts_array]) if existing_s.size > 0 else ts_array
            updated_m = np.concatenate([existing_m, tm_array]) if existing_m.size > 0 else tm_array

            # 保存更新后的数据
            np.save(x_savep, updated_x)
            np.save(y_savep, updated_y)
            np.save(s_savep, updated_s)
            np.save(m_savep, updated_m)

            print(f"Data from {name} saved successfully.")
        except Exception as e:
            print(f"Error saving data from {name}: {e}")

        # 记录已处理的文件名
        with open("processed_files.txt", "a") as f:
            f.write(name + "\n")
from concurrent.futures import ProcessPoolExecutor, as_completed
def creat_single_trainData1(dir, x_savep, y_savep, s_savep, m_savep, img_savep, img_p):
    global X, y, S, M, imgs, csv_names
    names = os.listdir(dir)
    # 计算需要选择的文件数量
    num_files_to_select = max(1, int(len(names) * psample))  # 确保至少选择一个文件
    random.seed(40)
    # 随机选取文件
    selected_csv_names = random.sample(names, num_files_to_select)



    if not os.path.exists(img_p):
        os.mkdir(img_p)

    # 使用线程池并行处理文件
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_single_file, name, dir, img_p, 32): name for name in selected_csv_names}
        for future in as_completed(futures):
            update_shared_lists1(future.result(),  x_savep, y_savep, s_savep, m_savep)

    print(csv_names)

"""
 12 24修改
"""
def interpolate_and_resample(df, target_length=64):
    # 插值前，确保数据按照时间戳排序
    df.sort_values(time_field, inplace=True)
    # 确保时间戳是浮点数
    df[time_field] = df[time_field].astype(float)
    # 去除'time'列中时间数值相同的行，只保留第一次出现的行
    df = df.drop_duplicates(subset=time_field, keep='first')
    
    # 使用三次样条插值填充缺失的数据点
    cs_lon = CubicSpline(df[time_field], df[lon_field])
    cs_lat = CubicSpline(df[time_field], df[lat_field])
    cs_vel = CubicSpline(df[time_field], df[sog_field])  # 注意：这里使用'sog_field'代表速度字段
    cs_cou = CubicSpline(df[time_field], df[cog_field])  # 注意：这里使用'cog_field'代表航向字段
    cs_vx = CubicSpline(df[time_field], df[vx_field])
    cs_vy = CubicSpline(df[time_field], df[vy_field])

    #cs_v1 = CubicSpline(df[time_field], df[v1_field])
    #cs_vx1 = CubicSpline(df[time_field], df[vx1_field])
    #cs_vy1 = CubicSpline(df[time_field], df[vy1_field])
    # 如果有其他需要插值的字段，可以在此处添加相应的CubicSpline对象

    # 生成新的时间戳序列，从最小时间戳到最大时间戳，步长为总时间跨度除以target_length
    time_range = np.linspace(df[time_field].min(), df[time_field].max(), target_length)

    # 使用三次样条插值函数计算新的经度和纬度值
    df_interpolated = pd.DataFrame({
        time_field: time_range,
        lon_field: cs_lon(time_range),
        lat_field: cs_lat(time_range),
        sog_field: cs_vel(time_range),  # 注意：这里使用'sog_field'作为速度的键
        cog_field: cs_cou(time_range),  # 注意：这里使用'cog_field'作为航向的键
        vx_field: cs_vx(time_range),
        vy_field: cs_vy(time_range),
        #v1_field: cs_v1(time_range),
        #vx1_field: cs_vx1(time_range),
        #vy1_field: cs_vy1(time_range)
        # 如果有其他需要插值的字段，可以在此处添加相应的字段
    })
    df_interpolated.index = range(len(df_interpolated))
    return df_interpolated
# 如果航迹大于规定，自适应下采样
def resample_to_64_points(df):
    # 确保航迹点按时间戳排序
    df.sort_values('time', inplace=True)

    # 计算相邻航迹点之间的距离
    # 将经纬度转换为numpy数组以便进行数学运算
    points = df[['lat', 'lon']].values
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)

    # 计算累积距离
    cumulative_distances = np.cumsum(distances)

    # 计算每个航迹点到起点的距离比例
    distance_ratios = cumulative_distances / cumulative_distances.max()

    # 生成64个均匀分布的采样点对应的距离比例
    sample_ratios = np.linspace(0, 1, 64)

    # 根据距离比例采样航迹点
    sample_indices = np.searchsorted(distance_ratios, sample_ratios * distance_ratios.max(), side='right')

    # 确保采样索引不超出航迹点的索引范围
    sample_indices = np.clip(sample_indices, 0, len(df) - 1).astype(int)

    # 根据采样索引选择航迹点
    sampled_df = df.iloc[sample_indices].reset_index(drop=True)
    sampled_df.index = range(len(sampled_df))
    return sampled_df


def csv2npy_withInter(path, img_savep, max_trace_len=32, max_num=30, isTimeStd=False):
    if os.path.getsize(path) < 50 * 1024:  # 50KB转换为字节
        return None, None, None, None, None, None
    

    f_name = [lat_field, lon_field, sog_field, base_date_time_field, cog_field, vx_field, vy_field]
    field_indices = [3, 4, 5, 2, 6, 8, 9]
    field_index_map = {field: index for field, index in zip(f_name, field_indices)}
    record_df = pd.DataFrame(columns=['file1', 'file2', 'label', 'scene'])

    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    if len(df[mmsi_field].unique()) < 10:
        return None, None, None, None, None, None

    # 提前计算所有航迹的 vx 和 vy，并添加到 DataFrame 中
    df[vx_field] = df[sog_field] * np.sin(np.radians(df[cog_field]))
    df[vy_field] = df[sog_field] * np.cos(np.radians(df[cog_field]))

    df[time_field] = df[time_field] - np.min(df[time_field])

    # def rolling_mean_filter(series, window_size=3):
    #     """对序列应用滚动均值滤波"""
    #     return series.rolling(window=window_size, center=True, min_periods=1).mean()

    # # 应用滚动均值滤波
    # df[vx_field] = df.groupby(mmsi_field)[vx_field].apply(lambda x: rolling_mean_filter(x)).reset_index(level=0, drop=True)
    # df[vy_field] = df.groupby(mmsi_field)[vy_field].apply(lambda x: rolling_mean_filter(x)).reset_index(level=0, drop=True)

    # 对每条航迹进行 V1、VX1 和 VY1 的计算
    mmsi_dfs = []
    for mmsi in df[mmsi_field].unique():
        sub_df = df[df[mmsi_field] == mmsi]
        if len(sub_df) >= 10:  # 只处理长度大于等于10的航迹
            mmsi_dfs.append(sub_df)

    # 合并所有处理后的航迹
    df = pd.concat(mmsi_dfs, ignore_index=True)

    # 初始化图像
    width, height = 224, 224
    image = np.zeros((height, width, 3), dtype=np.uint8)

    mmsi_prefix_map = {}

    # 第一步：生成所有可能的航迹对
    all_pairs = []  # 记录所有可能的航迹对及其标签
    for mmsi in df[mmsi_field].unique():
        sub_df = df[df[mmsi_field] == mmsi]
        sub_df.sort_values(by=base_date_time_field, inplace=True)
        sub_df.index = range(len(sub_df))

        if len(sub_df) < 10:
            continue

        sub_df_max_time = np.max(sub_df[base_date_time_field].values)
        sub_df_min_time = np.min(sub_df[base_date_time_field].values)

        for mmsi_2 in df[mmsi_field].unique():
            if mmsi_2 == mmsi:
                continue

            sub_df2 = df[df[mmsi_field] == mmsi_2]
            sub_df2.sort_values(by=base_date_time_field, inplace=True)
            sub_df2.index = range(len(sub_df2))

            if len(sub_df2) < 10:
                continue

            # 提取mmsi和mmsi_2的前缀部分
            mmsi_prefix = mmsi.split('_')[0]
            mmsi_2_prefix = mmsi_2.split('_')[0]

            sub_df2_max_time = np.max(sub_df2[base_date_time_field].values)
            sub_df2_min_time = np.min(sub_df2[base_date_time_field].values)

            # 时间段重叠或时间间隔过短/过长的跳过
            if not (sub_df_max_time < sub_df2_min_time or sub_df_min_time > sub_df2_max_time):
                continue

            if sub_df2_min_time - sub_df_max_time < 120 or sub_df2_min_time - sub_df_max_time > 7200:
                continue

            if mmsi_prefix == mmsi_2_prefix and sub_df2_min_time - sub_df_max_time < 300:
                continue

            # 判断是否为正样本
            label = 1 if mmsi_prefix == mmsi_2_prefix else 0
            all_pairs.append((mmsi, mmsi_2, label))

    # 第二步：筛选正样本和负样本
    positive_pairs = [pair for pair in all_pairs if pair[2] == 1]
    negative_pairs = [pair for pair in all_pairs if pair[2] == 0]

    # 按正样本10倍采样负样本
    sampled_negative_pairs = random.sample(negative_pairs, min(len(negative_pairs), 10 * len(positive_pairs)))

    # 合并正负样本
    selected_pairs = positive_pairs + sampled_negative_pairs
    random.shuffle(selected_pairs)  # 打乱顺序

    # 第三步：生成最终样本
    X = []
    y = []
    S = []
    M = []
    IMG_S = []

    for mmsi, mmsi_2, label in selected_pairs:
        sub_df = df[df[mmsi_field] == mmsi]
        sub_df2 = df[df[mmsi_field] == mmsi_2]

        sub_df_max_time = np.max(sub_df[base_date_time_field].values)
        sub_df_min_time = np.min(sub_df[base_date_time_field].values)

        sub_df2_max_time = np.max(sub_df2[base_date_time_field].values)
        sub_df2_min_time = np.min(sub_df2[base_date_time_field].values)
        # 调整顺序
        if np.max(sub_df[base_date_time_field].values) < np.max(sub_df2[base_date_time_field].values):
            track_1 = sub_df.drop_duplicates(subset=base_date_time_field, keep='first')
            track_2 = sub_df2.drop_duplicates(subset=base_date_time_field, keep='first')
        else:
            track_1 = sub_df2.drop_duplicates(subset=base_date_time_field, keep='first')
            track_2 = sub_df.drop_duplicates(subset=base_date_time_field, keep='first')

        # 插值和重采样
        if len(track_1) >= max_trace_len:
            temp_1 = track_1[-max_trace_len:].loc[:, f_name].values
        else:
            temp_1 = interpolate_and_resample(track_1).loc[:max_trace_len - 1, f_name].values

        if len(track_2) >= max_trace_len:
            temp_2 = track_2[:max_trace_len].loc[:, f_name].values
        else:
            temp_2 = interpolate_and_resample(track_2).loc[:max_trace_len - 1, f_name].values
        #print(temp_1)
        # 添加到结果
        X.append([temp_1, temp_2])
        y.append(label)
        S.append(path)
        M.append(f"{mmsi}:{mmsi_2}")
        del sub_df, sub_df2, track_1, track_2, temp_1, temp_2
        gc.collect()
    return X, y, record_df, IMG_S, S, M

if __name__ == '__main__':
    csv_dir = r'D:\AISData\dma\big\test'
    # 原始数据路径
    temp_csv_dir = os.path.join(csv_dir, 'rot_env_data')

    # 设置保存路径
    x_savep = os.path.join(csv_dir, 'inter_x_fcy.npy')
    y_savep = os.path.join(csv_dir, 'inter_y_fcy.npy')
    s_savep = os.path.join(csv_dir, 'inter_s_fcy.npy')
    m_savep = os.path.join(csv_dir, 'inter_m_fcy.npy')
    img_savep = os.path.join(csv_dir, 'inter_imgs_fcy.npy')

    img_p = os.path.join(csv_dir, 'TSM_imgs')

    creat_single_trainData1(temp_csv_dir, x_savep, y_savep,s_savep, m_savep, img_savep, img_p)

