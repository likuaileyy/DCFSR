import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


# 定义目录路径
directory = r'D:\AISData\us\big\train'
source_directory = os.path.join(directory, 'scene')
destination_directory = os.path.join(directory, 'rot_env_data')
image_directory = os.path.join(directory, 'rot_env_img')

# 定义字段名称变量
mmsi_field = 'MMSI'
base_date_time_field = 'BaseDateTime'
lat_field = 'LAT'
lon_field = 'LON'
sog_field = 'SOG'
cog_field = 'COG'
heading_field = 'Heading'

# 如果目标目录不存在，则创建
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

def add_additional_cut_points(track, thre_time=1800, thre_pointnum=25, min_add_time=600, max_add_time=1800):
    """在满足条件下增加额外切割点"""
    cut_points = []
    
    start_index = track.index[0]
    endIndex = track.index[-1]

    end_time = track.loc[endIndex]['BaseDateTime'] # 替换成实际的结束时间
    
    while True:
        if start_index in track.index:
            start_time = track.loc[start_index]['BaseDateTime']

        segment = track.loc[start_index:]
        if((endIndex - start_index > thre_pointnum) & (end_time - start_time > thre_time)):
            # 计算随机时间差
            additional_time = random.randint(min_add_time, max_add_time)
            target_time = start_time + additional_time
            # 找到最接近target_time的索引
            closest_index = (segment[base_date_time_field] - target_time).abs().idxmin()

            if closest_index not in cut_points:  # 避免重复添加
                cut_points.append(closest_index)
                start_index = int(closest_index) + 1  # 更新起始索引
        else:
            break

    return sorted(set(cut_points))


def process_file(file_path, destination_directory, image_directory):
    """处理单个文件"""
    try:
        df = pd.read_csv(file_path, usecols=[mmsi_field, base_date_time_field, lat_field, lon_field, sog_field, cog_field, heading_field], on_bad_lines='skip')
        
        if df.empty or len(df) <= 1:
            print(f"文件 {os.path.basename(file_path)} 无有效数据或仅包含表头，将被跳过")
            return
        
        df[base_date_time_field] = pd.to_numeric(df[base_date_time_field], errors='coerce')
        df = df[df[base_date_time_field].notna()]

        new_tracks = []
        track_indices = {}
        plt.figure(figsize=(16, 16))
        
        for mmsi, group in df.groupby(mmsi_field):
            last_time = None
            cut_points = []
            point_count_since_last_cut = 0
            
            thre_time = 300
            thre_pointnum = 20
            
            for index, row in group.iterrows():
                current_time = float(row[base_date_time_field])
                
                if last_time is None:
                    last_time = current_time
                    continue
                
                time_diff = current_time - last_time
                
                if time_diff > thre_time:
                    cut_points.append(index)
                    point_count_since_last_cut = 0
                
                point_count_since_last_cut += 1
                last_time = current_time
            
            additional_cut_points = add_additional_cut_points(group)
            cut_points.extend(additional_cut_points)
            cut_points = sorted(set(cut_points))
            temp_cut_points = []

            cleaned_cut_points = []
            len_cut = len(cut_points)
            thre_cut = 2
            p_cut = thre_cut / max(len_cut, 10e-5)
            if cut_points:
                start_index = group.index[0]
                end_index = group.index[-1]
                last_point = start_index
                for current_point in cut_points:
                    if ((current_point - last_point > thre_pointnum) & (current_point != 0) & (current_point + thre_pointnum < end_index)):
                        random_float = random.random()
                        if random_float < p_cut:
                            cleaned_cut_points.append(current_point)
                            last_point = current_point

            cut_points = cleaned_cut_points
            
            if cut_points:
                temp_tracks = []
                prev_index = 0
                for cp in cut_points:
                    segment = group.loc[prev_index:cp-1]
                    if not segment.empty:
                        temp_tracks.append(segment)
                    prev_index = cp
                last_segment = group.loc[prev_index:]
                if not last_segment.empty:
                    temp_tracks.append(last_segment)

                for temp_track in temp_tracks:
                    # 如果该段的数量大于 40，则去掉前 10 个点
                    if len(temp_track) > 50:
                        temp_track = temp_track.iloc[5:]  # 去掉前 10 个点

                    mmsi_str = str(int(mmsi)) if isinstance(mmsi, float) and mmsi.is_integer() else str(mmsi)
                    current_index = track_indices.get(mmsi_str, 0)
                    new_mmsi = f"{mmsi_str}_{current_index}"
                    track_indices[mmsi_str] = current_index + 1
                    temp_track[mmsi_field] = new_mmsi
                    new_tracks.append(temp_track)
            else:
                mmsi_str = str(int(mmsi)) if isinstance(mmsi, float) and mmsi.is_integer() else str(mmsi)
                new_mmsi = f"{mmsi_str}_{0}"
                group[mmsi_field] = new_mmsi
                new_tracks.append(group)

        combined_df = pd.concat(new_tracks, ignore_index=True)
        output_file_path = os.path.join(destination_directory, os.path.basename(file_path))
        combined_df.to_csv(output_file_path, index=False)

        for temp_track in new_tracks:
            plt.plot(temp_track[lon_field], temp_track[lat_field], label=f'MMSI: {temp_track[mmsi_field].iloc[0]}')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        image_file_path = os.path.join(image_directory, os.path.splitext(os.path.basename(file_path))[0] + ".png")
        plt.savefig(image_file_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def main():
    # 获取所有 CSV 文件
    files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith(".csv")]

    # 使用 ProcessPoolExecutor 并发处理
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_file, file, destination_directory, image_directory) for file in files]

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理过程中发生错误: {e}")

    print("处理完成！")


if __name__ == "__main__":
    main()