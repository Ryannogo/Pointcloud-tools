import pickle
import json
import numpy as np
import os
# 输入为step_curve_no_offset.pkl转化pkl文件中的各类数组为json文件

def convert_to_json_serializable(data):
    # 处理 NumPy 数组
    if isinstance(data, np.ndarray):
        return data.tolist()
    # 处理字典中的 NumPy 数组
    elif isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    # 处理列表中的 NumPy 数组
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    # 其他类型直接保留
    else:
        return data


def process_pkl_files(input_dir, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.pkl'):
            input_file_path = os.path.join(input_dir, filename)
            try:
                # 加载原始数据
                with open(input_file_path, "rb") as f:
                    data = pickle.load(f)

                # 转换数据
                json_data = convert_to_json_serializable(data)

                # 生成输出文件名
                output_filename = os.path.splitext(filename)[0] + '.json'
                output_file_path = os.path.join(output_dir, output_filename)

                # 保存为 JSON 文件
                with open(output_file_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                print(f"成功转换 {input_file_path} 到 {output_file_path}")
            except Exception as e:
                print(f"处理 {input_file_path} 时出错: {e}")


if __name__ == "__main__":
    input_directory = "NerVE-main/NerVE64Dataset/00001471"
    output_directory = "NerVE-main/json//00001471"
    process_pkl_files(input_directory, output_directory)
    
