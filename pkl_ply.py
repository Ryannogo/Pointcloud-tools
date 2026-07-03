import pickle
import os
import numpy as np

def pkl_to_ply(pkl_path, output_dir):
    """
    将.pkl文件中的曲线数据转换为PLY格式
    """
    # 读取.pkl数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 为每条曲线生成独立的PLY文件
    for curve_name, curve_data in data.items():
        pts = curve_data['points']
        edges = curve_data['edges']

        # 检查数据合法性
        if len(pts) < 2 or len(edges) == 0:
            print(f"跳过无效曲线: {curve_name} (点数: {len(pts)}, 边数: {len(edges)})")
            continue

        # 将numpy数组转换为Python原生类型（如果存在）
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        if isinstance(edges, np.ndarray):
            edges = edges.tolist()

        # 强制类型转换
        pts = [[float(x) for x in pt] for pt in pts]
        edges = [[int(i) for i in edge] for edge in edges]

        # 检查边索引越界
        max_index = len(pts) - 1
        valid_edges = []
        for edge in edges:
            if edge[0] < 0 or edge[0] > max_index or edge[1] < 0 or edge[1] > max_index:
                print(f"曲线 '{curve_name}' 的边索引越界: {edge} (最大索引: {max_index})")
                continue
            valid_edges.append(edge)

        # 写入PLY文件
        output_path = os.path.join(output_dir, f"{curve_name}.ply")
        with open(output_path, 'w') as f:
            # PLY头部信息
            f.write(
                f"ply\n"
                f"format ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                f"property float x\n"
                f"property float y\n"
                f"property float z\n"
                f"element edge {len(valid_edges)}\n"
                f"property int vertex1\n"
                f"property int vertex2\n"
                f"end_header\n"
            )

            # 写入顶点
            for pt in pts:
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

            # 写入边
            for edge in valid_edges:
                f.write(f"{edge[0]} {edge[1]}\n")

        print(f"已生成: {output_path}")

if __name__ == "__main__":
    pkl_path = "predict/00000010/cad_curves.pkl"
    output_dir = "predict/00000010/10"
    os.makedirs(output_dir, exist_ok=True)
    pkl_to_ply(pkl_path, output_dir)