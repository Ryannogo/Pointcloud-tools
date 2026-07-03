import os
import shutil

def copy_folders_by_list():
    # ===================== 【你只需要修改这里的3个路径】 =====================
    # 1. 存放文件名列表的 txt 文件路径
    list_txt_path = "NerVE-main/消融实验/目标文件.txt"
    
    # 2. 源文件夹（你要从这里找文件夹）
    source_folder = "NerVE-main/predict_N64D"
    
    # 3. 目标文件夹（复制到这里）
    target_folder = "NerVE-main/消融实验"
    # ======================================================================

    # 检查文件是否存在
    if not os.path.exists(list_txt_path):
        print(f"错误：列表文件不存在 → {list_txt_path}")
        return

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    # 读取列表
    with open(list_txt_path, "r", encoding="utf-8") as f:
        folder_names = [line.strip() for line in f if line.strip()]

    print(f"✅ 共读取到 {len(folder_names)} 个文件夹名称\n")

    # 开始复制
    success_count = 0
    fail_count = 0

    for folder_name in folder_names:
        src_path = os.path.join(source_folder, folder_name)
        dst_path = os.path.join(target_folder, folder_name)

        # 判断源文件夹是否存在
        if not os.path.isdir(src_path):
            print(f"❌ 不存在：{folder_name}")
            fail_count += 1
            continue

        # 复制文件夹（覆盖已存在的）
        try:
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)  # 先删除旧的
            shutil.copytree(src_path, dst_path)
            print(f"✅ 复制成功：{folder_name}")
            success_count += 1
        except Exception as e:
            print(f"⚠️ 复制失败：{folder_name}，原因：{str(e)}")
            fail_count += 1

    # 最终统计
    print("\n" + "="*50)
    print(f"任务完成：成功 {success_count} 个 | 失败/不存在 {fail_count} 个")
    print("="*50)

if __name__ == "__main__":
    copy_folders_by_list()