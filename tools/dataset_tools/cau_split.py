import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm  # 导入进度条库

def split_dataset(src_root, dst_root, val_ratio=0.2):
    sub_dirs = ['flood_vv', 'opt', 'vv']
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 1. 创建目标目录结构
    for s in ['train', 'val', 'test']:
        for sub in sub_dirs:
            (dst_root / s / sub).mkdir(parents=True, exist_ok=True)

    # 2. 处理 test 部分
    test_src_path = src_root / 'test' / sub_dirs[0]
    test_files = os.listdir(test_src_path)
    
    print("\n--- 正在处理 Test 集 (直接复制) ---")
    for file_name in tqdm(test_files, desc="Copying Test", unit="file"):
        for sub in sub_dirs:
            src_file = src_root / 'test' / sub / file_name
            dst_file = dst_root / 'test' / sub / file_name
            if src_file.exists():
                shutil.copy2(src_file, dst_file)

    # 3. 处理 train 部分：随机划分出 val
    train_src_path = src_root / 'train' / sub_dirs[0]
    train_files = os.listdir(train_src_path)
    
    random.seed(42)
    random.shuffle(train_files)

    val_count = int(len(train_files) * val_ratio)
    val_files = set(train_files[:val_count])
    
    print(f"\n--- 正在处理 Train/Val 集 (按 {val_ratio*100}% 比例划分) ---")
    # 使用 tqdm 包装循环
    for file_name in tqdm(train_files, desc="Splitting Train/Val", unit="file"):
        target_split = 'val' if file_name in val_files else 'train'
        
        for sub in sub_dirs:
            src_file = src_root / 'train' / sub / file_name
            dst_file = dst_root / target_split / sub / file_name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                print(f"\n[Warning] Missing: {src_file}")

    print(f"\n任务完成！")
    print(f"结果路径: {dst_root}")
    print(f"统计: 训练集 {len(train_files)-val_count} 张, 验证集 {val_count} 张, 测试集 {len(test_files)} 张")

if __name__ == "__main__":
    SOURCE_PATH = "/media/trifurs/备份盘/Download/Hete_CD/CAU"
    TARGET_PATH = "/media/trifurs/备份盘/Download/Hete_CD/CAU1"
    
    if os.path.exists(SOURCE_PATH):
        split_dataset(SOURCE_PATH, TARGET_PATH, val_ratio=0.2)
    else:
        print(f"错误：找不到源路径 {SOURCE_PATH}")
        