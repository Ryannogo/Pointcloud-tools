import numpy as np
import os
from plyfile import PlyData, PlyElement
from tqdm import tqdm  # 可选，用于显示进度条


def remove_duplicate_points_single(ply_input_path, ply_output_path, precision=16, verbose=False):
    """
    单个PLY文件去重（修复内存连续型错误）
    :param ply_input_path: 输入PLY文件路径
    :param ply_output_path: 输出去重后的PLY文件路径
    :param precision: 坐标精度（小数位数），避免浮点误差导致伪重复
    :param verbose: 是否打印单个文件的详细信息
    :return: (原始点数, 去重后点数, 是否成功)
    """
    try:
        # 1. 读取PLY文件并提取顶点数据
        ply_data = PlyData.read(ply_input_path)
        vertex = ply_data['vertex']
        
        # 2. 提取坐标（强制转为连续内存数组）
        x = np.ascontiguousarray(vertex['x'])
        y = np.ascontiguousarray(vertex['y'])
        z = np.ascontiguousarray(vertex['z'])
        points_coords = np.column_stack((x, y, z))  # 确保列拼接后内存连续
        original_count = len(points_coords)
        
        if original_count == 0:
            raise ValueError("文件中无顶点数据")
        
        # 3. 高效去重核心逻辑（修复内存连续问题）
        # 步骤1：舍入坐标避免浮点误差
        coords_rounded = np.round(points_coords, precision)
        # 步骤2：强制转为连续内存数组（关键修复！）
        coords_rounded = np.ascontiguousarray(coords_rounded)
        # 步骤3：创建结构化数组（替代view方法，更稳定）
        coords_struct = np.core.records.fromarrays(
            [coords_rounded[:,0], coords_rounded[:,1], coords_rounded[:,2]],
            names='x,y,z',
            formats='f8,f8,f8'
        )
        # 步骤4：找唯一坐标的索引
        _, unique_indices = np.unique(coords_struct, return_index=True)
        unique_indices = np.sort(unique_indices)  # 保持原始顺序
        unique_count = len(unique_indices)
        
        # 4. 提取唯一点的所有属性（保留RGB/法向量等）
        unique_vertex_data = {}
        for prop in vertex.properties:
            prop_name = prop.name
            # 对每个属性都确保内存连续
            prop_data = np.ascontiguousarray(vertex[prop_name])
            unique_vertex_data[prop_name] = prop_data[unique_indices]
        
        # 5. 构建新的PLY顶点数据
        dtype = []
        for prop in vertex.properties:
            prop_name = prop.name
            prop_dtype = vertex[prop_name].dtype
            dtype.append((prop_name, prop_dtype))
        
        # 组装结构化数组（再次确保连续）
        unique_points = np.empty(unique_count, dtype=dtype)
        for prop_name in unique_vertex_data.keys():
            unique_points[prop_name] = unique_vertex_data[prop_name]
        unique_points = np.ascontiguousarray(unique_points)
        
        # 6. 写入ASCII格式的PLY文件
        vertex_element = PlyElement.describe(unique_points, 'vertex')
        ply_out = PlyData([vertex_element], text=True)
        ply_out.write(ply_output_path)
        
        if verbose:
            print(f"  原始点数: {original_count} | 去重后点数: {unique_count} | 删除重复: {original_count - unique_count}")
        return (original_count, unique_count, True)
    
    except Exception as e:
        print(f"\n处理失败 {ply_input_path}: {str(e)}")
        return (0, 0, False)


def batch_remove_duplicates(input_dir, output_dir, recursive=False, precision=16, show_progress=True):
    """
    批量处理文件夹下的所有PLY文件
    :param input_dir: 输入文件夹路径（包含待处理的PLY文件）
    :param output_dir: 输出文件夹路径（保存去重后的文件）
    :param recursive: 是否递归遍历子文件夹（True=遍历所有子文件夹，False=仅单层）
    :param precision: 坐标精度（小数位数）
    :param show_progress: 是否显示进度条
    """
    # 1. 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：输入文件夹不存在 -> {input_dir}")
        return
    
    # 2. 创建输出文件夹（含子文件夹结构）
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 收集所有PLY文件路径
    ply_file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".ply"):
                # 构建输入/输出路径（保持子文件夹结构）
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subdir, file)
                ply_file_paths.append((input_path, output_path))
        
        # 如果不递归，只处理第一层就退出
        if not recursive:
            break
    
    # 4. 检查是否找到PLY文件
    if len(ply_file_paths) == 0:
        print(f"未在 {input_dir} 中找到任何PLY文件")
        return
    
    # 5. 批量处理文件
    total_files = len(ply_file_paths)
    success_count = 0
    total_original = 0
    total_unique = 0
    
    # 进度条（可选）
    iterator = tqdm(ply_file_paths, desc="批量处理PLY文件") if show_progress else ply_file_paths
    
    for input_path, output_path in iterator:
        original, unique, success = remove_duplicate_points_single(
            input_path, output_path, precision=precision, verbose=False
        )
        if success:
            success_count += 1
            total_original += original
            total_unique += unique
    
    # 6. 打印批量处理统计结果
    print("\n" + "="*50)
    print(f"批量处理完成！")
    print(f"总文件数: {total_files} | 成功处理: {success_count} | 失败: {total_files - success_count}")
    print(f"总原始点数: {total_original:,} | 总去重后点数: {total_unique:,} | 总删除重复点数: {total_original - total_unique:,}")
    print(f"输出文件夹: {output_dir}")
    print("="*50)


# ==================== 调用示例 ====================
if __name__ == "__main__":
    # 请修改以下路径为你的实际路径
    INPUT_FOLDER = r"EdgeFormer-master/EdgeFormer_dppi/datasets/bridge/label"  # 待处理的PLY文件夹
    OUTPUT_FOLDER = r"EdgeFormer-master/EdgeFormer_dppi/datasets/bridge/txt"  # 去重后的文件保存路径
    
    # 批量处理配置
    batch_remove_duplicates(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        recursive=False,  # 仅处理当前文件夹，不递归子文件夹
        precision=16,      # 坐标精度（6位小数足够）
        show_progress=True  # 显示进度条
    )

