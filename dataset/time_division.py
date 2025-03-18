import os
import pandas as pd
from datetime import timedelta
from multiprocessing import Pool, Manager

# 定义字段名称
TIMESTAMP = 'BaseDateTime'
MMSI = 'MMSI'
SOG = 'SOG'
COG = 'COG'
HEADING = 'Heading'
LAT = 'LAT'  # 假设纬度列为LAT
LON = 'LON'  # 假设经度列为LON

# 定义要遍历的目录和输出场景目录
base_dir = r'D:\AISData\us\big\src'
scene_output_dir = r'D:\AISData\us\big\scene'

# 确保输出场景目录存在
if not os.path.exists(scene_output_dir):
    os.makedirs(scene_output_dir)

def filter_small_range(group):
    if len(group) < 8:  # 如果航迹点少于8个，则过滤掉
        return None
    
    lat_min, lat_max = group[LAT].min(), group[LAT].max()
    lon_min, lon_max = group[LON].min(), group[LON].max()
    
    if (lat_max - lat_min) < 0.003 and (lon_max - lon_min) < 0.003:
        return None
    return group

def process_time_segment(time_segment_df, file_prefix, time_start, counter, lock):
    # 计算每个MMSI的经纬度运动区间，并过滤掉不符合条件的组
    filtered_groups = [group for _, group in time_segment_df.groupby(MMSI) if filter_small_range(group) is not None]
    if filtered_groups:  # 如果有符合条件的组
        filtered_df = pd.concat(filtered_groups, ignore_index=True)
    else:
        filtered_df = pd.DataFrame()  # 如果没有符合条件的组，则设置df为空DataFrame
    
    # 检查过滤后的数据框是否包含至少5个不同的MMSI
    if not filtered_df.empty and filtered_df[MMSI].nunique() >= 5:

        filtered_df = add_error_to_position(filtered_df)
        # 生成文件名：原文件名_开始时间.csv
        output_filename = f"{file_prefix}_{time_start.strftime('%Y-%m-%d_%H')}.csv"
        output_path = os.path.join(scene_output_dir, output_filename)
        filtered_df.to_csv(output_path, index=False)
        print(f"已保存: {output_path}")
        
        # 使用显式锁确保计数器修改是线程安全的
        with lock:
            counter.value += 1
        return True
    return False

import numpy as np
# 在process_file函数内，数据清洗和过滤完成后，但在保存文件之前的位置插入以下代码
def add_error_to_position(df):
    # 生成误差
    def generate_error():
        if np.random.choice([True, False]):
            return np.random.uniform(-0.03, -0.01)
        else:
            return np.random.uniform(0.01, 0.03)

    # 应用误差到LAT和LON列
    df[LAT] += df.apply(lambda row: generate_error(), axis=1)
    df[LON] += df.apply(lambda row: generate_error(), axis=1)
    
    return df

def process_file(input_path, counter, lock):
    if not os.path.isfile(input_path):
        print(f"文件 {input_path} 不存在。")
        return
    
    file_prefix = os.path.splitext(os.path.basename(input_path))[0]  # 获取文件名前缀（不带扩展名）
    
    df = pd.read_csv(input_path)
    
    # 时间戳转换，不指定具体格式让pandas自动识别
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])

    # 时间戳转换为秒级时间戳
    df[TIMESTAMP] = df[TIMESTAMP].astype('int64') // 10**9
    
    # 数据清洗：去除重复项并保留第一次出现的数据
    df = df.drop_duplicates(subset=[MMSI, TIMESTAMP], keep='first').reset_index(drop=True)
    
    # 按时间戳排序
    df = df.sort_values(by=TIMESTAMP).reset_index(drop=True)
    
    # 过滤无效值
    df = df[df[['SOG', 'COG', HEADING]].notna().all(axis=1) & (df['SOG'] > 1) & (df['COG'] < 360) & (df['COG'] > 0)]
    
    start_time = pd.to_datetime(df[TIMESTAMP].min(), unit='s')
    end_time = pd.to_datetime(df[TIMESTAMP].max(), unit='s')
    current_time = start_time
    
    while current_time < end_time:
        next_time = current_time + timedelta(hours=3)
        
        mask = (df[TIMESTAMP] >= int(current_time.timestamp())) & (df[TIMESTAMP] < int(next_time.timestamp()))
        time_segment_df = df[mask]
        
        if not time_segment_df.empty:
            process_time_segment(time_segment_df, file_prefix, current_time, counter, lock)
        
        current_time = next_time

# 主函数：使用多进程处理
def main():
    # 获取所有CSV文件的路径
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    # 使用Manager创建一个共享计数器和锁
    with Manager() as manager:
        counter = manager.Value('i', 1)  # 共享计数器，初始值为1
        lock = manager.Lock()  # 显式创建锁
        
        # 创建进程池，固定10个进程
        with Pool(processes=10) as pool:
            pool.starmap(process_file, [(file, counter, lock) for file in csv_files])

    print("所有文件处理完成。")

if __name__ == '__main__':
    main()