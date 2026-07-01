import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def rebalance_bright_dataset_v2(src_root, dst_root, ratio=(4, 1, 1)):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    # 定义子目录及其对应的文件后缀
    # 键是文件夹名，值是该文件夹下特有的后缀
    config = {
        'post-event': '_post_disaster.tif',
        'pre-event': '_pre_disaster.tif',
        'target': '_building_damage.tif'
    }
    
    old_splits = ['train', 'val', 'test']
    new_splits = ['train', 'val', 'test']

    # 1. 提取所有样本的“唯一 ID”
    # 我们以 target 文件夹作为基准来扫描 ID
    id_registry = {}  # {样本ID: 原本所在的 split}
    
    print("正在扫描数据集并匹配样本 ID...")
    for split in old_splits:
        target_path = src_root / split / 'target'
        if not target_path.exists():
            continue
            
        suffix = config['target']
        for file_name in os.listdir(target_path):
            if file_name.endswith(suffix):
                # 提取前缀：去掉 '_building_damage.tif'
                sample_id = file_name.replace(suffix, "")
                id_registry[sample_id] = split

    all_ids = list(id_registry.keys())
    total_count = len(all_ids)
    
    if total_count == 0:
        print("错误：未能在 target 文件夹中匹配到符合后缀的文件。请检查文件名！")
        return

    # 2. 随机打乱并按比例分配 ID
    random.seed(42)
    random.shuffle(all_ids)

    r_sum = sum(ratio)
    train_end = int(total_count * ratio[0] / r_sum)
    val_end = train_end + int(total_count * ratio[1] / r_sum)

    partition = {
        'train': all_ids[:train_end],
        'val': all_ids[train_end:val_end],
        'test': all_ids[val_end:]
    }

    # 3. 执行物理移动/复制
    print(f"\n--- 重新划分统计 (总样本 ID 数: {total_count}) ---")
    for s in new_splits:
        print(f"{s.capitalize()} 集: {len(partition[s])} 个样本")
        for sub in config.keys():
            (dst_root / s / sub).mkdir(parents=True, exist_ok=True)

    print("\n正在根据 ID 重新分发多模态文件...")
    for s_new, ids in partition.items():
        for sample_id in tqdm(ids, desc=f"Processing {s_new}", unit="sample"):
            s_old = id_registry[sample_id]
            
            # 根据 ID 和后缀还原出三个文件夹下各自的文件名
            for sub, suffix in config.items():
                file_name = f"{sample_id}{suffix}"
                src_file = src_root / s_old / sub / file_name
                dst_file = dst_root / s_new / sub / file_name
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    # 如果某些 ID 在某些文件夹下缺失，这里打印警告
                    print(f"\n[!] 缺失文件: {src_file}")

    print(f"\n✅ 处理完成！新数据集已保存在: {dst_root}")

if __name__ == "__main__":
    SOURCE = "/media/trifurs/备份盘/Download/Hete_CD/BRIGHT"
    TARGET = "/media/trifurs/备份盘/Download/Hete_CD/BRIGHT1"
    
    # 执行划分
    rebalance_bright_dataset_v2(SOURCE, TARGET, ratio=(4, 1, 1))
    