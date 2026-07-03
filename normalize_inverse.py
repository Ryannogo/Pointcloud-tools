import numpy as np
import trimesh
import os
from tqdm import tqdm

def inverse_pc_normalize(normalized_pc, center, scale, normalize_factor=0.9):
    """逆全局归一化：恢复原始坐标"""
    original_pc = (normalized_pc * scale) / normalize_factor + center
    return original_pc

def batch_inverse_normalize(input_dir, npz_root, output_dir):
    """
    批量逆归一化处理
    :param input_dir: 归一化后的PLY文件目录（例如：Cut50w_results/64_10/）
    :param npz_root:  参数文件根目录（例如：normalized/）
    :param output_dir: 恢复后的输出目录
    """
    # 收集所有PLY文件
    ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    if not ply_files:
        print(f"错误：输入目录中没有PLY文件！ {input_dir}")
        return

    print(f"发现 {len(ply_files)} 个待处理文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 进度条处理
    success = 0
    for ply_name in tqdm(ply_files, desc="逆归一化进度"):
        try:
            # 提取文件ID（假设文件名如 1.ply）
            file_id = os.path.splitext(ply_name)[0]
            
            # 构建文件路径
            ply_path = os.path.join(input_dir, ply_name)
            npz_path = os.path.join(npz_root, file_id, f"{file_id}.npz")
            
            # 加载数据
            norm_mesh = trimesh.load(ply_path)
            params = np.load(npz_path)
            
            # 参数检查
            if not all(k in params for k in ['center', 'scale']):
                raise ValueError("NPZ文件中缺少必要参数")
                
            # 执行逆归一化
            restored_pc = inverse_pc_normalize(
                norm_mesh.vertices,
                center=params['center'],
                scale=params['scale']
            )
            
            # 保存结果
            output_path = os.path.join(output_dir, f"restored_{ply_name}")
            trimesh.points.PointCloud(restored_pc).export(output_path)
            success += 1
            
        except Exception as e:
            print(f"\n处理失败: {ply_name}")
            print(f"错误类型: {type(e).__name__}, 详情: {str(e)}")

    print(f"\n处理完成！成功率: {success}/{len(ply_files)}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 配置路径
    input_dir = "Cut50/6410_5x"  # 包含所有归一化PLY的目录
    npz_root = "PCA_length_5x"             # 参数文件根目录（每个子目录包含对应npz）
    output_dir = "Cut50/6410_5x/restored"     # 恢复后的输出目录
    
    # 执行批量处理
    batch_inverse_normalize(input_dir, npz_root, output_dir)