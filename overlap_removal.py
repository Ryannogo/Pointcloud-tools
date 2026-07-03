import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import pickle

def read_point_cloud(file_path):
    """读取点云文件"""
    if file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
    elif file_path.endswith('.xyz') or file_path.endswith('.txt'):
        pcd = o3d.io.read_point_cloud(file_path, format='xyz')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return pcd

def detect_overlap_indices(pcd1, pcd2, threshold=0.01):
    """
    检测两个点云之间的重叠点索引
    
    参数:
    pcd1, pcd2: 要处理的两个点云
    threshold: 距离阈值，小于此值的点将被视为重叠点
    
    返回:
    第一个点云中重叠点的索引列表
    """
    # 计算两个点云之间的最近邻距离
    pcd_tree = o3d.geometry.KDTreeFlann(pcd2)
    points1 = np.asarray(pcd1.points)
    
    overlap_indices = []
    
    # 遍历第一个点云中的每个点，检查是否在第二个点云的阈值范围内
    for i in tqdm(range(len(points1)), desc="Detecting overlaps"):
        point = points1[i]
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if k > 0 and np.linalg.norm(point - np.asarray(pcd2.points)[idx[0]]) < threshold:
            overlap_indices.append(i)
    
    return overlap_indices

def overlap_removal(input_dir,output_dir,start_idx,end_idx,threshold,voxel_size):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成要处理的文件列表
    file_list = []
    for i in range(start_idx, end_idx + 1):
        file_name = f"restored_{i:08d}.ply"
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            file_list.append(file_path)
        else:
            print(f"Warning: File {file_path} does not exist, skipping.")
    
    if not file_list:
        print("No files found for processing.")
        return
    
    # 阶段1: 检测所有相邻点云对之间的重叠点
    print("\n=== Phase 1: Detecting overlaps between adjacent point clouds ===")
    overlap_map = {}  # 存储每个点云需要移除的点的索引
    
    # 为每个点云初始化空的重叠索引列表
    for i in range(len(file_list)):
        overlap_map[i] = set()
    
    # 检测相邻点云之间的重叠
    for i in range(len(file_list) - 1):
        print(f"\nAnalyzing overlap between {os.path.basename(file_list[i])} and {os.path.basename(file_list[i+1])}")
        pcd1 = read_point_cloud(file_list[i])
        pcd2 = read_point_cloud(file_list[i+1])
        
        # 检测pcd1中与pcd2重叠的点
        indices1 = detect_overlap_indices(pcd1, pcd2, threshold=threshold)
        overlap_map[i].update(indices1)
        
        # 检测pcd2中与pcd1重叠的点
        indices2 = detect_overlap_indices(pcd2, pcd1, threshold=threshold)
        # 由于是相邻点云对，我们只需要考虑后一个点云的重叠部分
        overlap_map[i+1].update(indices2)
    
    # 保存重叠点索引信息，以便后续分析或调试
    overlap_info_file = os.path.join(output_dir, "overlap_indices.pkl")
    with open(overlap_info_file, 'wb') as f:
        pickle.dump(overlap_map, f)
    print(f"\nOverlap information saved to: {overlap_info_file}")
    
    # 阶段2: 合并所有点云并移除重叠点
    print("\n=== Phase 2: Merging point clouds and removing overlaps ===")
    merged_pcd = o3d.geometry.PointCloud()
    total_points_before = 0
    
    # 合并所有点云，同时记录每个点云在合并后的点云中的起始索引
    start_indices = []
    for i, file_path in enumerate(tqdm(file_list, desc="Merging point clouds")):
        pcd = read_point_cloud(file_path)
        start_idx = len(merged_pcd.points)
        start_indices.append(start_idx)
        merged_pcd += pcd
        total_points_before += len(pcd.points)
    
    print(f"Total points before overlap removal: {len(merged_pcd.points)}")
    
    # 计算全局重叠点索引（在合并后的点云中的索引）
    global_overlap_indices = set()
    for i, indices in overlap_map.items():
        if indices:
            start_idx = start_indices[i]
            global_indices = {start_idx + idx for idx in indices}
            global_overlap_indices.update(global_indices)
    
    print(f"Number of points to remove: {len(global_overlap_indices)}")
    
    # 移除重叠点
    if global_overlap_indices:
        points = np.asarray(merged_pcd.points)
        colors = np.asarray(merged_pcd.colors) if merged_pcd.has_colors() else None
        
        # 创建保留点的索引
        keep_indices = np.ones(len(points), dtype=bool)
        keep_indices[list(global_overlap_indices)] = False
        
        # 创建处理后的点云
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_indices])
        if colors is not None and len(colors) > 0:
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[keep_indices])
        
        merged_pcd = filtered_pcd
    
    print(f"Total points after overlap removal: {len(merged_pcd.points)}")
    
    # 可选：下采样以减少点云密度
    if voxel_size > 0:
        print(f"Downsampling with voxel size: {voxel_size}")
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Points after downsampling: {len(merged_pcd.points)}")
    
    # 保存最终结果
    output_file = os.path.join(output_dir, "merged_0.01.ply")
    o3d.io.write_point_cloud(output_file, merged_pcd)
    print(f"\nSuccessfully merged {len(file_list)} point clouds!")
    print(f"Final result saved to: {output_file}")

if __name__ == "__main__":
    # === 在这里设置你的参数 ===
    input_dir = "Cut50/6410_5x/restored"  # 输入点云文件夹路径
    output_dir = "Cut50/6410_5x/restored"          # 输出结果文件夹路径
    start_idx = 1                           # 起始点云编号
    end_idx = 10                            # 结束点云编号
    threshold = 0.01                         # 重叠检测阈值（关键参数，需要调整）
    voxel_size = 0.0                        # 下采样体素大小（0表示不进行下采样）
    overlap_removal(input_dir,output_dir,start_idx,end_idx,threshold,voxel_size)
    