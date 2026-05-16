import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


class FinetuneTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device

    def training(self, epochs=30, lr_base=5e-6, lr_head=2e-4, patience=15, save_path="best_model.pt"):
        # 提取真实模型以获取参数
        if isinstance(self.model, torch.nn.DataParallel):
            real_model = self.model.module
        else:
            real_model = self.model

        base_params = [p for n, p in real_model.named_parameters() if "encoder" in n and p.requires_grad]
        head_params = [p for n, p in real_model.named_parameters() if "encoder" not in n and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': lr_base},
            {'params': head_params, 'lr': lr_head}
        ], weight_decay=1e-2, fused=False)

        best_metric = 0.0
        early_stop_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            steps = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    # 兼容多卡：如果 loss 是多维的（每张卡一个），取平均值
                    if loss.dim() > 0:
                        loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 验证
            metrics = self.evaluate(self.valid_loader)
            print(
                f"Ep {epoch + 1} | Loss:{epoch_loss / steps:.4f} | AUC:{metrics['auc']:.4f} | AUPRC:{metrics['auprc']:.4f}")

            # 保存逻辑
            if metrics['auprc'] > best_metric:
                best_metric = metrics['auprc']
                early_stop_counter = 0

                state_to_save = real_model.state_dict()
                torch.save(state_to_save, save_path)
                print(f">>>  Best Valid AUPRC: {best_metric:.4f} | Model saved to {save_path}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f" Early Stopping at Epoch {epoch + 1}")
                    break

        print(f"Loading best weights from {save_path} for final evaluation...")
        real_model.load_state_dict(torch.load(save_path))
        return best_metric

    def evaluate(self, loader):
        self.model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = self.model(**batch)
                    logits = output["logits"]

                    p = torch.softmax(logits, dim=-1)[:, 1] if logits.size(-1) == 2 else torch.sigmoid(
                        logits.squeeze(-1))

                probs.extend(p.cpu().float().numpy())
                labels.extend(batch["labels"].cpu().float().numpy())

        try:
            auc = roc_auc_score(labels, probs)
            auprc = average_precision_score(labels, probs)
        except:
            auc, auprc = 0, 0

        return {"auc": auc, "auprc": auprc, "mcc": 0, "threshold": 0.5}