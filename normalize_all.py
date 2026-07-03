import numpy as np
import trimesh
import os
from scipy.spatial import KDTree
from tqdm import tqdm

def pc_normalize(pc, normalize_factor=0.9):
    """全局归一化: 缩放到 [-0.9, 0.9]"""
    bmin, bmax = np.min(pc, axis=0), np.max(pc, axis=0)
    center = (bmin + bmax) / 2.0
    scale = np.max(bmax - bmin) / 2.0 + 1e-8
    normalized_pc = (pc - center) * (normalize_factor / scale)
    return normalized_pc, center, scale


def network_input_normalize(pc, knn_size=8, grid_size=64, mode='cube_face'):
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

def process_single_file(input_path, output_root):
    """处理单个子目录中的PLY文件"""
    try:
        # 获取父目录名称作为ID
        file_id = os.path.basename(os.path.dirname(input_path))
        
        # 读取点云
        mesh = trimesh.load(input_path)
        if not hasattr(mesh, 'vertices'):
            raise ValueError("无效的点云文件")
        pc = mesh.vertices
        
        # 执行全局归一化
        pc_norm, center, scale = pc_normalize(pc)
        
        # 创建输出目录
        output_dir = os.path.join(output_root, file_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存归一化点云
        norm_mesh = trimesh.points.PointCloud(pc_norm)
        norm_mesh.export(os.path.join(output_dir, f"{file_id}.ply"))
        
        # 保存参数文件
        np.savez(
            os.path.join(output_dir, f"{file_id}.npz"),
            center=center,
            scale=scale,
            grid_size=64
        )
        return True
    
    except Exception as e:
        print(f"\n处理失败: {input_path}")
        print(f"错误信息: {str(e)}")
        return False

def batch_process(input_root, output_root):
    """批量处理主函数"""
    # 收集所有子目录中的PLY文件
    task_list = []
    for subdir in os.listdir(input_root):
        subdir_path = os.path.join(input_root, subdir)
        if os.path.isdir(subdir_path):
            # 查找PLY文件
            for file in os.listdir(subdir_path):
                if file.endswith(".ply"):
                    task_list.append(os.path.join(subdir_path, file))
                    break  # 每个子目录只处理第一个PLY文件
    
    if not task_list:
        print("未找到需要处理的PLY文件！")
        return
    
    print(f"共发现 {len(task_list)} 个子目录需要处理")
    
    # 使用进度条处理
    success = 0
    for path in tqdm(task_list, desc="处理进度"):
        if process_single_file(path, output_root):
            success += 1
    
    print(f"\n处理完成！成功处理 {success}/{len(task_list)} 个子目录")
    print(f"输出结构示例: {output_root}/00000001/00000001.ply")
    print(f"参数文件示例: {output_root}/00000001/00000001.npz")

if __name__ == "__main__":
    # 配置参数
    input_root = "NerVE-main/Cut50/DGCNN"    # 原始数据根目录
    output_root = "NerVE-main/Cut50/DGCNN"   # 输出根目录
    
    # 执行批量处理
    batch_process(input_root, output_root)