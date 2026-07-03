#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将PLY文件表头中的所有"double"数据类型替换为"float"
支持：
1. 处理单个PLY文件
2. 批量处理指定文件夹下的所有PLY文件
3. 保留原文件备份（可选）
"""

from pathlib import Path
import shutil


def replace_double_with_float_in_ply(ply_input_path: Path, ply_output_path: Path):
    """
    处理单个PLY文件，替换表头中的double为float
    
    Args:
        ply_input_path: 输入PLY文件路径
        ply_output_path: 输出修改后的PLY文件路径
    """
    try:
        # 读取文件所有行
        with open(ply_input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified_lines = []
        header_ended = False
        
        # 逐行处理：只修改表头部分的double
        for line in lines:
            # 标记表头结束
            if line.strip().lower() == 'end_header':
                header_ended = True
                modified_lines.append(line)
                continue
            
            # 表头部分：替换double为float（不区分大小写，比如Double/Double也能处理）
            if not header_ended:
                modified_line = line.replace('double', 'float').replace('Double', 'Float').replace('DOUBLE', 'FLOAT')
                modified_lines.append(modified_line)
            # 非表头部分：原样保留
            else:
                modified_lines.append(line)
        
        # 写入修改后的内容
        with open(ply_output_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        
        print(f"[成功] 处理完成：{ply_input_path.name} -> {ply_output_path}")
        
    except Exception as e:
        print(f"[失败] 处理{ply_input_path.name}出错：{str(e)}")


def batch_process_ply_folder(
    input_dir: Path,
    output_dir: Path,
    keep_original_backup: bool = False,
    backup_suffix: str = '_original'
):
    """
    批量处理文件夹下的所有PLY文件
    
    Args:
        input_dir: 存放原始PLY文件的文件夹
        output_dir: 存放修改后PLY文件的文件夹
        keep_original_backup: 是否为原文件创建备份
        backup_suffix: 备份文件的后缀（比如xxx_original.ply）
    """
    # 创建输出文件夹（如果不存在）
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有PLY文件
    ply_files = list(input_dir.glob("*.ply"))
    if not ply_files:
        print(f"[提示] 在{input_dir}中未找到任何PLY文件")
        return
    
    print(f"[信息] 找到{len(ply_files)}个PLY文件，开始批量处理...")
    
    for ply_file in ply_files:
        # 输出文件路径（和原文件同名）
        output_file = output_dir / ply_file.name
        
        # 可选：创建原文件备份
        if keep_original_backup:
            backup_file = input_dir / f"{ply_file.stem}{backup_suffix}{ply_file.suffix}"
            shutil.copy2(ply_file, backup_file)
            print(f"[备份] 已创建原文件备份：{backup_file.name}")
        
        # 处理单个文件
        replace_double_with_float_in_ply(ply_file, output_file)
    
    print(f"[完成] 批量处理结束，共处理{len(ply_files)}个文件，结果保存在：{output_dir}")


if __name__ == "__main__":
    # ====================== 请根据你的需求修改以下配置 ======================
    # 处理模式：single（单文件） / batch（批量）
    PROCESS_MODE = "batch"
    
    # -------- 单文件处理配置 --------
    SINGLE_INPUT_PLY = Path("path/to/your/input.ply")  # 你的输入PLY文件路径
    SINGLE_OUTPUT_PLY = Path("path/to/your/output.ply")  # 修改后的保存路径
    
    # -------- 批量处理配置 --------
    BATCH_INPUT_DIR = Path("EdgeFormer-master/EdgeFormer_dppi/datasets/bridge/ply_vertex copy")  # 原始PLY文件夹
    BATCH_OUTPUT_DIR = Path("EdgeFormer-master/EdgeFormer_dppi/datasets/bridge/ply_vertex")  # 修改后保存的文件夹
    KEEP_BACKUP = True  # 是否保留原文件备份
    # =======================================================================
    
    if PROCESS_MODE == "single":
        # 处理单个文件
        if not SINGLE_INPUT_PLY.exists():
            print(f"错误：输入文件{SINGLE_INPUT_PLY}不存在！")
        else:
            replace_double_with_float_in_ply(SINGLE_INPUT_PLY, SINGLE_OUTPUT_PLY)
    
    elif PROCESS_MODE == "batch":
        # 批量处理文件夹
        if not BATCH_INPUT_DIR.exists():
            print(f"错误：输入文件夹{BATCH_INPUT_DIR}不存在！")
        else:
            batch_process_ply_folder(
                input_dir=BATCH_INPUT_DIR,
                output_dir=BATCH_OUTPUT_DIR,
                keep_original_backup=KEEP_BACKUP
            )
    
    else:
        print("错误：PROCESS_MODE只能是single或batch！")