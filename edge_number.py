import os
import pickle

def load_folder_list(all_txt_path):
    """加载all.txt中的文件夹列表"""
    with open(all_txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_single_folder(root_dir, folder_name):
    """处理单个目标文件夹"""
    # 构建完整路径
    pkl_path = os.path.join(root_dir, folder_name, "step_edge_reso64.pkl")
    output_path = os.path.join(root_dir, folder_name, f"{folder_name}.txt")
    
    # 检查文件存在性
    if not os.path.exists(pkl_path):
        print(f"警告：跳过文件夹 {folder_name}，未找到pkl文件")
        return False
    
    try:
        # 提取edge编号
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 过滤并排序edge键
        edge_keys = [k for k in data.keys() if k.startswith('edge')]
        sorted_numbers = sorted(
            (int(k[4:]) for k in edge_keys if k[4:].isdigit()),
            key=lambda x: x
        )
        
        # 写入文件
        with open(output_path, 'w') as f:
            f.write('\n'.join(map(str, sorted_numbers)))
        
        print(f"成功生成：{output_path}")
        return True
    
    except Exception as e:
        print(f"处理 {folder_name} 失败：{str(e)}")
        return False

def batch_processing(root_dir, all_txt_path):
    """批量处理主函数"""
    # 加载目标文件夹列表
    try:
        folders = load_folder_list(all_txt_path)
    except FileNotFoundError:
        print(f"错误：未找到all.txt文件 {all_txt_path}")
        return
    
    # 验证输入参数
    if not os.path.isdir(root_dir):
        print(f"错误：根目录不存在 {root_dir}")
        return
    
    # 处理每个文件夹
    success_count = 0
    for folder in folders:
        if process_single_folder(root_dir, folder):
            success_count += 1
    
    print(f"处理完成，成功 {success_count}/{len(folders)} 个文件夹")

# 使用示例（需替换实际路径）
if __name__ == '__main__':
    ROOT_DIR = "NerVE64Dataset"  # 所有目标文件夹的父目录
    ALL_TXT = "NerVE64Dataset/all.txt"      # 存储目标文件夹名的清单文件
    
    batch_processing(ROOT_DIR, ALL_TXT)