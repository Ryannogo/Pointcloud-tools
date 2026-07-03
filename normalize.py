import numpy as np
import trimesh
import os
from scipy.spatial import KDTree

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

#======================== 主函数 ========================#
if __name__ == "__main__":
    # 输入输出设置
    input_ply = "bridg_part/00000001/拱肋_label.ply"  # 替换为你的点云PLY路径
    output_dir = "normalized"
    grid_size = 64
    mode = 'cube_face'  # 或 'geom'

    # 读取点云 (直接获取顶点，忽略面)
    pointcloud = trimesh.load(input_ply)
    pc = pointcloud.vertices  # 直接获取顶点坐标，形状为 (N, 3)

    # 执行归一化
    result = network_input_normalize(pc, knn_size=8, grid_size=grid_size, mode=mode)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存归一化后的点云 (仅顶点)
    pc_norm_mesh = trimesh.points.PointCloud(vertices=result['pc_norm'])
    pc_norm_mesh.export(os.path.join(output_dir, "normalize_label.ply"))
    
    # 保存参数
    np.savez(
        os.path.join(output_dir, "00000001.npz"),
        center=result['center'],
        scale=result['scale'],
        grid_size=result['grid_size'],
    )
    print("处理完成！输出目录:", output_dir)