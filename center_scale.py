import numpy as np
import trimesh
import os
from scipy.spatial import KDTree
import glob

def pc_normalize(pc, normalize_factor=0.9):
    """全局归一化: 缩放到 [-0.9, 0.9]"""
    bmin, bmax = np.min(pc, axis=0), np.max(pc, axis=0)
    center = (bmin + bmax) / 2.0
    scale = np.max(bmax - bmin) / 2.0 + 1e-8
    normalized_pc = (pc - center) * (normalize_factor / scale)
    return normalized_pc, center, scale

def network_input_normalize(pc, knn_size=8, grid_size=64, mode='cube_face'):
    """局部归一化: 生成模型输入格式"""
    # 全局归一化
    pc_norm, center, scale = pc_normalize(pc)
    
    # 计算 KNN 邻域
    tree = KDTree(pc_norm)
    _, knn_idx = tree.query(pc_norm, k=knn_size)
    
    if mode == 'cube_face':
        knn_pos = pc_norm[knn_idx] - pc_norm[:, np.newaxis, :]
        knn_pos *= grid_size
    elif mode == 'geom':
        step = 2.0 / grid_size
        pc_grid_idx = np.floor((pc_norm + 1) / step).astype(int)
        centers = pc_grid_idx * step + (step / 2.0 - 1.0)
        knn_pos = pc_norm[knn_idx] - centers[:, np.newaxis, :]
        knn_pos *= grid_size
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return {
        'pc_norm': pc_norm,
        'knn_pos': knn_pos,
        'knn_idx': knn_idx,
        'center': center,
        'scale': scale,
        'grid_size': grid_size,
    }

#======================== 批量处理函数 ========================#
def process_raw_to_flat_output(raw_dir, output_dir, knn_size=8, grid_size=64, mode='cube_face'):
    """
    批量处理raw目录下的PLY文件，直接输出到指定文件夹
    输入结构: raw_dir/00000000/pc_obj.ply
    输出结构: output_dir/00000000.npz
    
    :param raw_dir: 输入根目录 (raw文件夹)
    :param output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有符合模式的PLY文件
    ply_pattern = os.path.join(raw_dir, "*", "pc_obj.ply")
    ply_files = glob.glob(ply_pattern)
    
    if not ply_files:
        print(f"警告: 在 {raw_dir} 下未找到符合条件的PLY文件")
        return
    
    # 遍历所有找到的PLY文件
    for file_path in ply_files:
        try:
            # 解析路径结构，获取子文件夹名称作为输出文件名
            # 例如: raw/00000000/pc_obj.ply -> 输出为00000000.npz
            folder_name = os.path.basename(os.path.dirname(file_path))
            
            # 读取点云
            pointcloud = trimesh.load(file_path)
            pc = pointcloud.vertices
            
            # 执行归一化
            result = network_input_normalize(pc, knn_size=knn_size, grid_size=grid_size, mode=mode)
            
            # 生成输出文件路径（直接输出到output_dir，用文件夹名作为文件名）
            output_path = os.path.join(output_dir, f"{folder_name}.npz")
            
            # 保存为NPZ文件
            np.savez(
                output_path,
                center=result['center'],
                scale=result['scale'],
                grid_size=result['grid_size'],
                pc_norm=result['pc_norm'],
                knn_pos=result['knn_pos'],
                knn_idx=result['knn_idx']
            )
            
            print(f"处理成功: {file_path} -> {output_path}")
            
        except Exception as e:
            print(f"处理失败 {file_path}: {str(e)}")

#======================== 主函数 ========================#
if __name__ == "__main__":
    # 输入输出设置
    raw_dir = "NerVE-main/Cut50/DGCNN"  # 输入根目录，即raw文件夹
    output_dir = "output"  # 直接输出到output文件夹
    grid_size = 64
    knn_size = 8
    mode = 'cube_face' 

    # 批量处理所有符合条件的PLY文件
    process_raw_to_flat_output(
        raw_dir=raw_dir,
        output_dir=output_dir,
        knn_size=knn_size,
        grid_size=grid_size,
        mode=mode
    )
    
    print("批量处理完成！")
    print(f"输入根目录: {raw_dir}")
    print(f"输出目录: {output_dir}")