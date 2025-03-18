import pandas as pd
import os
import math
from multiprocessing import Process, Manager
import gc

data_dir = r'D:\AISData\us\big'  # 包含多个CSV文件的文件夹
save_dir = os.path.join(data_dir, 'src')

# 如果保存目录不存在，则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 分区函数
def get_partition(lat, lon, delta=2):
    lat0 = math.floor(lat / delta) * delta
    lat1 = lat0 + delta
    lon0 = math.floor(lon / delta) * delta
    lon1 = lon0 + delta
    return lat0, lat1, lon0, lon1

latname = 'LAT'
lonname = 'LON'

# 处理单个CSV文件的函数
def process_file(filepath, save_dir, file_prefix):
    print(f"正在处理文件: {filepath}")
    
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    batch_size = 10000  # 每批次处理的数据行数
    for start in range(0, len(df), batch_size):  # 按批次处理数据
        end = start + batch_size
        partitioned_data = {}
        
        for index, row in df.iloc[start:end].iterrows():
            lat0, lat1, lon0, lon1 = get_partition(row[latname], row[lonname])
            key = f"{int(lat0)}_{int(lat1)}_{int(lon0)}_{int(lon1)}"
            
            if key not in partitioned_data:
                partitioned_data[key] = []
            partitioned_data[key].append(row)
        
        for key, rows in partitioned_data.items():
            partition_df = pd.DataFrame(rows)
            process_partition(partition_df, save_dir, file_prefix)
        
        partitioned_data.clear()
        gc.collect()  # 手动触发垃圾回收

# 处理每个分区的数据，并写入CSV
def process_partition(rows, save_dir, file_prefix):
    if not rows.empty:
        row = rows.iloc[0]  # 使用分区中的任意一行获取经纬度信息
        lat0, lat1, lon0, lon1 = get_partition(row[latname], row[lonname])
        key = f"index_{int(lat0)}_{int(lat1)}_{int(lon0)}_{int(lon1)}"
        file_path = os.path.join(save_dir, f"{file_prefix}_{key}.csv")
        rows.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))

# 主函数：启动多进程处理
def main():
    processes = []
    
    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            file_prefix = os.path.splitext(filename)[0]  # 获取文件名前缀
            
            # 创建一个进程来处理该文件
            p = Process(target=process_file, args=(filepath, save_dir, file_prefix))
            processes.append(p)
            p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()

    print("所有文件分区完成并保存为CSV文件。")

if __name__ == '__main__':
    main()