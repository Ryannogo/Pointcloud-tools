import pickle
import numpy as np
import os
from plyfile import PlyData, PlyElement

def pkl_to_ply(pkl_path, ply_path, verbose=True):
    """
    直接将pkl点云文件转换为ply格式
    :param pkl_path: 输入pkl文件路径
    :param ply_path: 输出ply文件路径
    :param verbose: 是否打印处理进度
    """
    try:
        # 读取pkl文件
        if verbose: 
            print(f"正在读取 {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 提取点云数据 pc or points
        if 'points' not in data:
            raise KeyError("pkl文件中未找到'points'字段,或者字段选择错误")
            

        points = np.asarray(data['points'], dtype=np.float32)
        if verbose:
            print(f"成功读取 {points.shape[0]} 个点 | 坐标维度: {points.shape[1]}")

        # 验证坐标维度
        if points.shape[1] not in [3, 6]:
            print("警告: 非常规坐标维度 (支持3D坐标或带RGB的6D坐标)")

        # 创建PLY数据结构
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if points.shape[1] == 6:
            dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
        vertex = np.array([tuple(p) for p in points], dtype=dtype)
        vertex_element = PlyElement.describe(vertex, 'vertex')

        # 写入PLY文件
        if verbose:
            print(f"正在生成 {ply_path}...")
        PlyData([vertex_element], text=True).write(ply_path)
        
        if verbose:
            print(f"转换完成！已保存至 {ply_path}")

    except Exception as e:
        print(f"转换失败: {str(e)}")
        raise

# if __name__ == "__main__":
#     # 使用示例
#     input_pkl = "Cut50w/00000002/pred_64_pwl_curve.pkl"  # 替换为你的pkl文件路径
#     output_ply = "Cut50w/00000002/2.ply"      # 输出文件路径
    
#     pkl_to_ply(input_pkl, output_ply)

# #批量处理
# if __name__ == "__main__":
#     # 读取all.txt中的编号
#     with open("NerVE-main/NerVE64Dataset/HD_error.txt", "r") as f:#******************************
#         num_strs = [line.strip() for line in f if line.strip()] 
    
#     for num_str in num_strs:
#         # 构建输入和输出路径 Cut50/64_5x,pred_64_10_pwl.pklNerVE-main/NerVE64Dataset
#         input_pkl = os.path.join("NerVE-main","NerVE64Dataset", num_str, "pc_obj.pkl")#****************
#         output_ply = os.path.join("NerVE-main","Cut50","DGCNN","pc_obj",f"{(num_str)}.ply")  # 转换为数字去除前导零，下次试试把num_str前的int去掉是不是就保留0了*************

        
#         # 执行转换
#         if os.path.exists(input_pkl):
#             pkl_to_ply(input_pkl, output_ply)
#             print(f"成功转换: {input_pkl} -> {output_ply}")
#         else:
#             print(f"文件不存在，跳过: {input_pkl}")
# # #NerVE64Dataset/00000003/step_curve_no_offset.pkl

# ===================== 【批量处理：遍历所有pkl文件】 =====================
if __name__ == "__main__":
    # ========== 只需要改这 3 个路径 ==========
    # 你的编号列表文件
    txt_file_path = "NerVE-main/消融实验/目标文件.txt"
    
    # 源根目录：NerVE-main/消融实验/00000976/xxx.pkl
    source_root = "NerVE-main/消融实验"
    
    # 输出ply的根目录
    output_root = "NerVE-main/消融实验/消融实验_ply"
    # ========================================

    # 读取编号列表
    with open(txt_file_path, "r") as f:
        num_strs = [line.strip() for line in f if line.strip()] 

    print(f"✅ 读取到 {len(num_strs)} 个文件夹，开始遍历转换所有pkl...\n")

    # 遍历每个编号文件夹
    for num_str in num_strs:
        # 当前文件夹路径
        folder_path = os.path.join(source_root, num_str)
        
        # 文件夹不存在则跳过
        if not os.path.isdir(folder_path):
            print(f"❌ 文件夹不存在：{folder_path}")
            continue

        # ============= 核心：遍历当前文件夹里 所有 .pkl 文件 =============
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
        
        if len(pkl_files) == 0:
            print(f"ℹ️ {num_str} 文件夹内无pkl文件，跳过")
            continue

        print(f"\n📂 进入 {num_str}，找到 {len(pkl_files)} 个pkl文件")

        # 逐个转换
        for pkl_file in pkl_files:
            # 输入路径
            input_pkl = os.path.join(folder_path, pkl_file)
            
            # 输出路径：保持文件名不变，只改后缀为 .ply
            ply_name = os.path.splitext(pkl_file)[0] + ".ply"
            # 输出结构：output_root/00000976_pred_dk40_pwl.ply
            output_ply = os.path.join(output_root, f"{num_str}_{ply_name}")

            # 自动创建输出目录
            os.makedirs(output_root, exist_ok=True)

            # 转换
            pkl_to_ply(input_pkl, output_ply, verbose=False)
            print(f"  ✅ {pkl_file} → {ply_name}")

    print("\n🎉🎉🎉 全部转换完成！")