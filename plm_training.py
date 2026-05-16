import torch
import os
import random
import numpy as np
from transformers import T5Tokenizer
from dataset import make_plm_dataloader
from models.interaction_model import ProTCR_InducedFit_Final
from trainers import FinetuneTrainer


def set_seed(seed=1996):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    set_seed(1996)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    data_dir = "/tmp/pycharm_project_932/data/HS"
    model_path = "Rostlab/prot_t5_xl_uniref50"

    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    fold_scores = []

    for fold in range(1, 6):
        print(f"\n{'=' * 40}\n### Processing Fold {fold} ###\n{'=' * 40}")

        train_file = os.path.join(data_dir, f"{fold}_1_1train.csv")
        test_file = os.path.join(data_dir, f"{fold}_1_1test.csv")

        train_loader, valid_loader, _ = make_plm_dataloader(
            train_file=train_file, valid_file=test_file, test_file=test_file,
            tokenizer=tokenizer, batch_size=128
        )

        print(">> Init Model (12 Layers Finetune)...")
        model = ProTCR_InducedFit_Final(
            model_path=model_path,
            project_dim=256,
            unfreeze_last_n_layers=12
        )
        model = model.to(device)

        print(">> Compiling model...")
        try:

            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: Compile failed ({e}), running in eager mode.")

        trainer = FinetuneTrainer(model, train_loader, valid_loader, None, device)


        current_save_path = f"/tmp/pycharm_project_932/output/HS_best_model_fold{fold}.pt"


        best_auprc = trainer.training(
            epochs=50,
            lr_base=5e-6,
            lr_head=1e-5,
            patience=15,
            save_path=current_save_path
        )
        fold_scores.append(best_auprc)

        # 清理显存
        del model, trainer, train_loader, valid_loader
        torch.cuda.empty_cache()

    print(f"\n🏆 Final 5-Fold AUPRC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()