import torch
import os
import numpy as np
import pandas as pd  # 用于保存预测分数
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score  # [新增] 导入 roc_auc_score
)
from transformers import T5Tokenizer

from dataset import make_plm_dataloader
from models.interaction_model import ProTCR_InducedFit_Final


def evaluate_model(model, loader, device):
    """独立的评估函数，用于计算测试集指标并返回分数"""
    model.eval()
    probs, labels_list, preds = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            # 将数据移动到设备上
            batch = {k: v.to(device) for k, v in batch.items()}

            # 开启混合精度推理，节省显存并加速
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(**batch)
                logits = output["logits"]
                # 提取正类的概率作为预测分数
                p = torch.softmax(logits, dim=-1)[:, 1]

            probs.extend(p.cpu().float().numpy())
            preds.extend((p > 0.5).long().cpu().numpy())
            labels_list.extend(batch["labels"].cpu().float().numpy())

    # 计算各项评估指标
    try:
        auc = roc_auc_score(labels_list, probs)  # [新增] 计算 ROC AUC
        auprc = average_precision_score(labels_list, probs)
        precision = precision_score(labels_list, preds, zero_division=0)
        recall = recall_score(labels_list, preds, zero_division=0)
        f1 = f1_score(labels_list, preds, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(labels_list, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    except Exception as e:
        print(f"Warning: Metric calculation failed ({e})")
        # [修改] 异常处理中加入 auc 的默认值
        auc, auprc, precision, specificity, recall, f1 = 0, 0, 0, 0, 0, 0

    metrics = {
        "auc": auc,  # [新增] 存入 metrics 字典
        "auprc": auprc,
        "precision": precision,
        "specificity": specificity,
        "recall": recall,
        "f1": f1
    }

    return metrics, probs, labels_list


def main():
    # 1. 基础设置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data_dir = "/tmp/pycharm_project_932/data/AS"
    model_path = "Rostlab/prot_t5_xl_uniref50"


    weights_path = "/tmp/pycharm_project_932/output/best_model_fold5.pt"

    test_files = [
        "1_1_1independent_unseen.csv"
        # "1_1_1independent_test.csv"
        # "1_1_1test.csv"
    ]

    # 2. 初始化分词器和模型架构
    print(">> Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    print(">> Initializing Model Architecture...")
    model = ProTCR_InducedFit_Final(
        model_path=model_path,
        project_dim=256,
        unfreeze_last_n_layers=12
    )

    # 3. 加载训练好的权重
    print(f">> Loading Weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            # 截取掉前面的 '_orig_mod.' (长度为10)
            new_key = k[10:]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)


    # 4. 循环测试每个数据集
    for test_file_name in test_files:
        print(f"\n{'=' * 50}")
        print(f"🚀 Testing on Dataset: {test_file_name}")
        print(f"{'=' * 50}")

        test_file_path = os.path.join(data_dir, test_file_name)

        _, _, test_loader = make_plm_dataloader(
            train_file=test_file_path,
            valid_file=test_file_path,
            test_file=test_file_path,
            tokenizer=tokenizer,
            batch_size=64
        )

        # 运行评估，接收指标和分数
        metrics, probs, labels = evaluate_model(model, test_loader, device)

        # [修改] 更新输出排版以包含 AUC
        print("\n📊 Evaluation Results:")
        print(f"{'AUC':<12} | {'AUPRC':<12} | {'Precision':<12} | {'Specificity':<12} | {'Recall':<12} | {'F1':<12}")
        print("-" * 88)
        print(
            f"{metrics['auc']:<12.4f} | {metrics['auprc']:<12.4f} | {metrics['precision']:<12.4f} | {metrics['specificity']:<12.4f} | {metrics['recall']:<12.4f} | {metrics['f1']:<12.4f}\n")

        output_csv = f"pred_scores_{test_file_name}"
        df_scores = pd.DataFrame({
            'True_Label': labels,
            'Predicted_Probability': probs
        })
        df_scores.to_csv(output_csv, index=False)
        print(f"✅ Predicted scores and labels have been successfully saved to: {output_csv}\n")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()