import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader


# 1. 读取数据 - 保持不变 (逻辑很棒)
def read_data(file):
    if file.endswith('csv') or file.endswith('txt'):
        # 自动处理 csv 和 txt (制表符)
        df = pd.read_csv(file, sep=r'[,\t]+', engine="python")
    elif file.endswith('xlsx'):
        df = pd.read_excel(file)
    else:
        raise NotImplementedError(f'Does not support file type: {file}')

    # 清理列名
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "")

    # 检查列
    required_cols = ['Epitope', 'CDR3B', 'Affinity']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"文件 {file} 缺少列: {missing_cols}。当前列名: {df.columns.tolist()}")

    # 提取数据
    peptide = df['Epitope'].astype(str).str.strip().tolist()
    cdr3 = df['CDR3B'].astype(str).str.strip().tolist()
    label = df['Affinity'].fillna(0).astype(int).tolist()

    return peptide, cdr3, label


class PLMDataset(Dataset):
    def __init__(self, peptide, cdr3, labels):
        self.peptide = peptide
        self.cdr3 = cdr3
        self.labels = labels

    def __getitem__(self, index):
        return self.peptide[index], self.cdr3[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# 2. 整理器 - 【优化点：对齐 A100 Tensor Core】
class Collator:
    def __init__(self, tokenizer, peptide_max_len=24, cdr3_max_len=32):
        self.tokenizer = tokenizer
        # 稍微放宽一点长度，防止特殊 Token (</s>) 溢出
        self.peptide_max_len = peptide_max_len
        self.cdr3_max_len = cdr3_max_len

    def __call__(self, batch):
        peptide, cdr3, labels = list(zip(*batch))

        # 预处理：ProtT5 需要加空格
        def clean(seq):
            s = str(seq).upper().strip()
            for char in [' ', '\t', '"', "'"]:
                s = s.replace(char, "")
            return " ".join(list(s))

        # 【优化】pad_to_multiple_of=8
        # A100 的 Tensor Core 计算 8 的倍数维度时效率最高
        p_inputs = self.tokenizer(
            [clean(p) for p in peptide],
            padding="max_length",  # 保持 max_length 对 torch.compile 更友好
            truncation=True,
            max_length=self.peptide_max_len,
            pad_to_multiple_of=8,  # <--- 新增
            return_tensors='pt'
        )

        c_inputs = self.tokenizer(
            [clean(c) for c in cdr3],
            padding="max_length",
            truncation=True,
            max_length=self.cdr3_max_len,
            pad_to_multiple_of=8,  # <--- 新增
            return_tensors='pt'
        )

        return {
            "pep_ids": p_inputs['input_ids'],
            "pep_mask": p_inputs['attention_mask'],
            "tcr_ids": c_inputs['input_ids'],
            "tcr_mask": c_inputs['attention_mask'],
            "labels": torch.LongTensor(labels)
        }


# 3. 构造 Loader - 【优化点：多进程与内存锁页】
def make_plm_dataloader(train_file, valid_file, test_file, tokenizer, batch_size=128):
    # 读取数据
    train_pep, train_cdr, train_label = read_data(train_file)
    valid_pep, valid_cdr, valid_label = read_data(valid_file)
    test_pep, test_cdr, test_label = read_data(test_file)

    train_dataset = PLMDataset(train_pep, train_cdr, train_label)
    valid_dataset = PLMDataset(valid_pep, valid_cdr, valid_label)
    test_dataset = PLMDataset(test_pep, test_cdr, test_label)

    collator = Collator(tokenizer=tokenizer)

    # 【核心优化配置】
    # 如果是在 Linux (A100) 上，num_workers 建议 8 或 16
    # 如果是在 Windows 上调试，num_workers 必须为 0
    num_workers = 8 if os.name != 'nt' else 0

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collator,
        "num_workers": num_workers,  # 多进程加载
        "pin_memory": True,  # 锁页内存，传输到 GPU 更快
        "persistent_workers": True if num_workers > 0 else False,  # 保持子进程不销毁
        "prefetch_factor": 2 if num_workers > 0 else None,  # 每个 worker 提前加载 2 个 batch
    }

    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    # 验证集和测试集不需要 shuffle，也不一定需要 persistent_workers (但开了也没事)
    valid_dataloader = DataLoader(valid_dataset, **loader_kwargs)
    test_dataloader = DataLoader(test_dataset, **loader_kwargs)

    return train_dataloader, valid_dataloader, test_dataloader