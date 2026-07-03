import os

# 原文件路径

def fold_transfer(original_path):
# 步骤1: 遍历路径下的所有文件并进行重命名
    for root, dirs, files in os.walk(original_path):
        for file in files:
            # 获取文件的完整路径
            old_file_path = os.path.join(root, file)
            # 提取文件名（不包含扩展名）
            file_name_without_extension = os.path.splitext(file)[0]
            # 新的文件名
            new_file_name = file_name_without_extension + "/TraMeth.txt"
            # 新的文件路径
            new_file_path = os.path.join(os.path.dirname(old_file_path), new_file_name)
            # 创建新的目录（如果不存在）
            new_dir = os.path.dirname(new_file_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
    # 步骤2: 收集所有文件夹名称并保存到all.txt
    all_folders = []
    for item in os.listdir(original_path):
        item_path = os.path.join(original_path, item)
        if os.path.isdir(item_path):
            all_folders.append(item)

    # 保存到all.txt
    all_txt_path = os.path.join(original_path, "all.txt")
    with open(all_txt_path, "w") as f:
        f.write("\n".join(all_folders))

    print(f"所有文件夹名称已保存到: {all_txt_path}")
if __name__ == "__main__":
    
    original_path = "traditional method/results32"
    fold_transfer(original_path)
