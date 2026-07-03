# import torch
# from grid_pooling import avg_pooling_forward

# # 创建 GPU 张量（显式指定 dtype=torch.int32）
# point_feat = torch.randn(100000, 512).cuda()
# pidx = torch.randint(0, 1000, (100000,), dtype=torch.int32).cuda()          # 关键修改
# pidx_counts = torch.randint(1, 10, (1000,), dtype=torch.int32).cuda()       # 关键修改

# # 调用前向传播
# out = avg_pooling_forward(point_feat, pidx, pidx_counts, 10)
# print(out.device)  # 应输出 "cuda:0"

# import pickle
# import numpy as np

# def inspect_pkl(pkl_path):
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
    
#     # 打印数据类型
#     print(f"数据类型: {type(data)}")
    
#     # 如果是字典，打印键和值形状
#     if isinstance(data, dict):
#         print("字典结构:")
#         for key in data:
#             value = data[key]
#             if isinstance(value, (np.ndarray, list)):
#                 print(f"  Key: {key}, Shape: {np.array(value).shape}")
#             else:
#                 print(f"  Key: {key}, Type: {type(value)}")
    
#     # 如果是numpy数组
#     elif isinstance(data, np.ndarray):
#         print(f"数组形状: {data.shape}, 数据类型: {data.dtype}")
    
#     # 示例：查看部分数据
#     print("\n示例数据片段:")
#     if isinstance(data, dict) and 'points' in data:
#         print("前5个点坐标:\n", data['points'][:5])
#     if isinstance(data, dict) and 'edges' in data:
#         print("前100条边连接:\n", data['edges'][:100])

# # 使用示例
# inspect_pkl("NerVE_dataset/00022787/step_curve_no_offset.pkl")

import pickle
import numpy as np
# 假设数据文件为 data.pkl
with open("ModelNet40/2/cad_curves.pkl", "rb") as f:
    data = pickle.load(f)

# # 查看顶层结构
# print("Top-level keys:", data.keys())

# # 查看顶点结构
# print("\nVertices shape and type:", type(data["vertices"]), data["vertices"].shape)

# # 查看边结构示例
# sample_edge = data["edges"][0]
# print("\nEdge keys:", sample_edge.keys())
# print("Edge is_closed:", sample_edge["is_closed"])
# print("Edge type:", sample_edge["type"])
# print("Edge samples shape:", sample_edge["samples"].shape)


# for curve_name, curve_data in data.items():
#     # 新增数据结构验证
#     if not isinstance(curve_data, dict):
#         print(f"曲线 {curve_name} 数据格式错误，预期字典，实际得到{type(curve_data)}")
#         continue
#     if 'points' not in curve_data or 'edges' not in curve_data:
#         print(f"曲线 {curve_name} 缺少必要字段")
#         continue

# 在加载数据后添加
endpoints_data = data.get('endpoints', None)
print("[调试] endpoints 类型:", type(endpoints_data))
if hasattr(endpoints_data, 'shape'):  # 如果是 numpy 数组
    print("[调试] endpoints 形状:", endpoints_data.shape)
elif isinstance(endpoints_data, (list, dict)):
    print("[调试] 前2个端点示例:", endpoints_data[:2] if isinstance(endpoints_data, list) else list(endpoints_data.items())[:2])

from plyfile import PlyData
from plyfile import PlyElement
def save_endpoints_as_ply(endpoints, output_path):
    """
    将端点保存为只有顶点的 PLY 文件
    """
    if endpoints is None:
        return
    
    # 确保是 numpy 数组
    if not isinstance(endpoints, np.ndarray):
        endpoints = np.array(endpoints)

    # 创建 PLY 对象
    verts = endpoints.reshape(-1, 3)
    ply = PlyData([
        PlyElement.describe(verts, 'vertex', comments=['端点数据'])
    ])

    # 保存文件
    ply.write(output_path)
    print(f"端点文件已保存至: {output_path} (含 {len(verts)} 个点)")