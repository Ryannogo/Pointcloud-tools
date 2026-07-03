import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


# ========================
# 数据加载
# ========================

def load_ply_samples(ply_path):
    """PLY → Open3D → numpy"""
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    return np.asarray(pcd.points)


def load_step_samples(data_path, offset=None):
    """读取 STEP 曲线采样"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    pts = data['points']
    edges = data['edges']

    midpts = np.mean(pts[np.asarray(edges)], axis=1)
    samples = np.concatenate([pts, midpts], axis=0)

    if offset is not None:
        samples += offset

    return samples


# ========================
# AABB 对齐
# ========================

def align_by_aabb(pred, gt):
    pred_center = (pred.min(axis=0) + pred.max(axis=0)) / 2
    gt_center = (gt.min(axis=0) + gt.max(axis=0)) / 2
    return pred + (gt_center - pred_center)


# ========================
# 指标计算
# ========================

def calc_metrics(pred, gt, threshold=0.01, max_HD=False):

    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)

    dist_pred2gt, _ = tree_gt.query(pred)
    dist_gt2pred, _ = tree_pred.query(gt)

    chamfer_dist = np.mean(dist_pred2gt**2) + np.mean(dist_gt2pred**2)

    bhaussdorf_dist = (
        max(dist_pred2gt.max(), dist_gt2pred.max())
        if max_HD
        else (dist_pred2gt.max() + dist_gt2pred.max()) / 2
    )

    TP = np.sum(dist_pred2gt < threshold)
    FP = len(dist_pred2gt) - TP
    FN = np.sum(dist_gt2pred >= threshold)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        'CD': chamfer_dist,
        'BHD': bhaussdorf_dist,
        'Precision': precision,
        'Recall': recall,
        'IoU': iou
    }


# ========================
# 主流程
# ========================

if __name__ == "__main__":

    pred_root = "traditional method/EdgeFormer"
    gt_root = "NerVE-main/NerVE64Dataset"

    input_list = "NerVE-main/NerVE64Dataset/val.txt"
    result_file = "NerVE-main/EVAL/eval_edgeAABB_val01.txt"

    distance_threshold = 0.01

    with open(input_list, "r") as f:
        target_folders = [line.strip() for line in f if line.strip()]

    with open(result_file, "w") as f:
        f.write("Folder_ID\tCD\tBHD\tPrecision\tRecall\tIoU\n")

    processed_count = 0

    for folder in target_folders:

        if not (folder.isdigit() and len(folder) == 8):
            continue

        pred_path = os.path.join(pred_root, folder, "edgeformer.ply")
        gt_curve_path = os.path.join(gt_root, folder, "nerve_reso64_curve.pkl")
        gt_step_path = os.path.join(gt_root, folder, "step_curve_no_offset.pkl")

        if not all(os.path.exists(p) for p in [pred_path, gt_curve_path, gt_step_path]):
            print(f"{folder} 缺少文件，跳过")
            continue

        try:
            # STEP offset
            with open(gt_curve_path, "rb") as f:
                offset = pickle.load(f)["stable_offset"]

            # 加载数据
            pred_samples = load_ply_samples(pred_path)
            gt_samples = load_step_samples(gt_step_path, offset)

            # AABB 对齐
            pred_samples = align_by_aabb(pred_samples, gt_samples)

            metrics = calc_metrics(
                pred_samples,
                gt_samples,
                threshold=distance_threshold
            )

            with open(result_file, "a") as f:
                f.write(
                    f"{folder}\t"
                    f"{metrics['CD']:.8e}\t"
                    f"{metrics['BHD']:.8e}\t"
                    f"{metrics['Precision']:.6f}\t"
                    f"{metrics['Recall']:.6f}\t"
                    f"{metrics['IoU']:.6f}\n"
                )

            processed_count += 1

        except Exception as e:
            print(f"{folder} 出错: {e}")

    print(f"完成：{processed_count}/{len(target_folders)}")
