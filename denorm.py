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
    :param input_dir: 归一化后的PLY文件目录（文件直接存放在此）
    :param npz_root: 参数文件目录（NPZ文件直接存放在此）
    :param output_dir: 恢复后的输出目录
    """
    # 获取所有PLY文件并提取ID
    ply_files = [f for f in os.listdir(input_dir) 
                if os.path.isfile(os.path.join(input_dir, f)) 
                and f.endswith('.ply') 
                and len(os.path.splitext(f)[0]) == 8 
                and os.path.splitext(f)[0].isdigit()]
    
    if not ply_files:
        print(f"错误：输入目录中没有符合要求的PLY文件！ {input_dir}")
        return
    
    print(f"发现 {len(ply_files)} 个待处理文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 进度条处理
    success = 0
    for ply_file in tqdm(ply_files, desc="逆归一化进度"):
        try:
            # 提取ID（文件名，不含扩展名）
            file_id = os.path.splitext(ply_file)[0]
            
            # 构建输入PLY文件路径
            ply_path = os.path.join(input_dir, ply_file)
            
            # 构建NPZ文件路径
            npz_path = os.path.join(npz_root, f"{file_id}.npz")
            
            # 检查NPZ文件是否存在
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"未找到对应的NPZ文件: {npz_path}")
            
            # 加载数据
            norm_mesh = trimesh.load(ply_path)
            params = np.load(npz_path)
            
            # 参数检查
            required_keys = ['center', 'scale']
            if not all(k in params for k in required_keys):
                missing = [k for k in required_keys if k not in params]
                raise ValueError(f"NPZ文件中缺少必要参数: {', '.join(missing)}")
                
            # 执行逆归一化
            restored_pc = inverse_pc_normalize(
                norm_mesh.vertices,
                center=params['center'],
                scale=params['scale']
            )
            
            # 保存到输出目录
            output_path = os.path.join(output_dir, f"{file_id}.ply")
            trimesh.points.PointCloud(restored_pc).export(output_path)
            success += 1
            
        except Exception as e:
            print(f"\n处理失败: {ply_file}")
            print(f"错误类型: {type(e).__name__}, 详情: {str(e)}")

    print(f"\n处理完成！成功率: {success}/{len(ply_files)}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 配置路径 - 根据你的实际路径修改
    input_dir = "NerVE-main/bridge/ply"  # PLY文件直接存放的目录
    npz_root = "NerVE-main/bridge/npz"   # NPZ文件直接存放的目录
    output_dir = "NerVE-main/bridge/restored"  # 恢复后的输出目录
    
    # 执行批量处理
    batch_inverse_normalize(input_dir, npz_root, output_dir)
