import pickle
import numpy as np
import os
from plyfile import PlyData, PlyElement
import pywt
import open3d as o3d


def wavelet_denoise(points, wavelet="db1", level=1, threshold=0.05):
    """
    简单小波去噪: 对点云坐标逐维进行离散小波变换，抑制高频噪声
    """
    denoised = []
    for i in range(points.shape[1]):
        coeffs = pywt.wavedec(points[:, i], wavelet, level=level)
        coeffs_thresh = [pywt.threshold(c, threshold * np.max(c)) for c in coeffs]
        rec = pywt.waverec(coeffs_thresh, wavelet)
        denoised.append(rec[:points.shape[0]])
    return np.vstack(denoised).T.astype(np.float32)


def pkl_to_ply(pkl_path, ply_path, verbose=True, denoise_method=None):
    """
    直接将pkl点云文件转换为ply格式 (支持可选去噪)
    :param pkl_path: 输入pkl文件路径
    :param ply_path: 输出ply文件路径
    :param verbose: 是否打印处理进度    :param denoise_method: 去噪方法 [None, 'statistical', 'radius', 'wavelet']
    """
    try:
        # 读取pkl文件
        if verbose:
            print(f"正在读取 {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if 'pc' not in data:
            raise KeyError("pkl文件中未找到'points'字段")

        points = np.asarray(data['pc'], dtype=np.float32)
        if verbose:
            print(f"成功读取 {points.shape[0]} 个点 | 坐标维度: {points.shape[1]}")

        # 去噪处理
        if denoise_method is not None:
            if verbose:
                print(f"正在应用去噪方法: {denoise_method}")

            if denoise_method == "wavelet":
                points = wavelet_denoise(points)

            else:
                # 用 open3d 做邻域去噪
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])

                if denoise_method == "statistical":
                    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                elif denoise_method == "radius":
                    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

                points = np.asarray(pcd.points)

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


if __name__ == "__main__":
    with open("NerVE-main/NerVE64Dataset/4362.txt", "r") as f:
        num_strs = [line.strip() for line in f if line.strip()]
#NerVE-main/Cut50/DGCNN/pc_obj
#
    for num_str in num_strs:
        input_pkl = os.path.join("NerVE-main","NerVE64Dataset", num_str, "pc_obj.pkl")
        output_ply = os.path.join("NerVE-main","Cut50", "val_ply", f"{num_str}.ply")

        if os.path.exists(input_pkl):
            # ⭐ 在这里可以选择去噪方法: None / "statistical" / "radius" / "wavelet"
            pkl_to_ply(input_pkl, output_ply, denoise_method=None)
            print(f"成功转换: {input_pkl} -> {output_ply}")
        else:
            print(f"文件不存在，跳过: {input_pkl}")
