import torch
import numpy as np
import pandas as pd
import os  # 1. 导入os模块
from models.DLS import classifier
from utils.one_trial import LitModel
from utils.prepare_data import get_Ksucc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, matthews_corrcoef,
                             confusion_matrix, recall_score, average_precision_score)
from torch.utils.data import Subset
import yaml
import time
from datetime import datetime

# 记录开始时间
start_time = time.time()
start_datetime = datetime.now()
print(f"🚀 集成训练开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 加载配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 加载数据
trainset, testset = get_Ksucc()
labels = trainset.y
neg_count = np.sum(labels == 0)
pos_count = np.sum(labels == 1)
total_count = neg_count + pos_count
w0 = total_count / (2 * neg_count) if neg_count > 0 else 1.0
w1 = total_count / (2 * pos_count) if pos_count > 0 else 1.0
class_weights = torch.tensor([w0, w1], dtype=torch.float)
print(f"数据不平衡：负样本数={neg_count}, 正样本数={pos_count}, 类权重 w0={w0:.4f}, w1={w1:.4f}")

# 超参数
hparams = classifier.get_hparams()
hparams['class_weights'] = class_weights
model_params = classifier.get_model_params()

def KCV():
    # 2. 创建本次实验专属的结果文件夹
    output_dir = f"result/{config['name']}result"
    os.makedirs(output_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [trainset[i][1] for i in range(len(trainset))]
    results_list = []

    test_proba_list = []
    test_labels = [testset[i][1] for i in range(len(testset))]

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainset, labels)):
        fold_start_time = time.time()
        print(f"\n🔄 Training fold {fold + 1}/5...")

        # 初始化模型
        model = LitModel(classifier, hparams, model_params)
        model.set_ckptdir("ckpt-model")
        train_subset = Subset(trainset, train_idx)
        val_subset = Subset(trainset, val_idx)

        model.fit(train_subset, valid_data=val_subset)

        # === 测试集预测 ===
        test_proba_tensor = model.predict_proba(testset)
        test_proba = test_proba_tensor.cpu().numpy().flatten()
        test_proba_list.append(test_proba)
        val_proba_tensor = model.predict_proba(val_subset)
        val_proba = val_proba_tensor.cpu().numpy().flatten()
        val_true = [val_subset[i][1] for i in range(len(val_subset))]
        val_metrics = model.test(val_subset)
        val_proba_for_metrics = val_proba_tensor.cpu().numpy().flatten()
        val_metrics['AP'] = average_precision_score(val_true, val_proba_for_metrics)
        results_list.append(val_metrics)
        fold_duration = time.time() - fold_start_time
        print(f"✅ Fold {fold + 1} 完成，耗时: {fold_duration // 60:.0f}分{fold_duration % 60:.1f}秒")

    # === Soft Voting ===
    avg_proba = np.mean(test_proba_list, axis=0)
    soft_pred = (avg_proba >= 0.5).astype(int)

    # === 保存评估结果 ===
    def save_preds(name, preds, probas=None):
        df = pd.DataFrame({'true': test_labels, 'pred': preds})
        df.to_csv(f"{output_dir}/{name}_ensemble_result.csv", index=False)
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)
        mcc = matthews_corrcoef(test_labels, preds)
        sn = recall_score(test_labels, preds)

        tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        bacc = (sn + sp) / 2
        auc_score, aupr, ap_score = None, None, None
        if probas is not None:
            auc_score = roc_auc_score(test_labels, probas)
            precision, recall, _ = precision_recall_curve(test_labels, probas)
            aupr = auc(recall, precision)
            ap_score = average_precision_score(test_labels, probas)

        print(f"\n{name} 评估结果：")
        print(f"Recall(SN): {sn:.4f}")
        print(f"Specificity(SP): {sp:.4f}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"AUC       : {auc_score:.4f}" if auc_score is not None else "AUC       : N/A")
        print(f"AP        : {ap_score:.4f}" if ap_score is not None else "AP        : N/A")
        print(f"MCC       : {mcc:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"AUPR      : {aupr:.4f}" if aupr is not None else "AUPR      : N/A")
        print(f"bacc      : {bacc:.4f}")

        metrics = {
            'SN': sn, 'SP': sp, 'Accuracy': acc, 'AUC': auc_score,
            'AP': ap_score, 'MCC': mcc, 'F1': f1, 'AUPR': aupr, 'bacc': bacc
        }
        pd.DataFrame([metrics]).to_csv(f"{output_dir}/{name}_metrics.csv", index=False)

    save_preds("soft", soft_pred, avg_proba)

    results_df = pd.DataFrame(results_list)
    mean_values = results_df.mean()
    std_values = results_df.std()
    summary_df = pd.DataFrame([mean_values, std_values], index=["Mean", "Std"])
    results_df = pd.concat([results_df, summary_df])
    formatted_row = {col: f"{mean_values[col]:.4f}±{std_values[col]:.4f}" for col in mean_values.index}
    results_df.loc["Mean±Std"] = formatted_row
    results_df.to_csv(f"{output_dir}/5fold_validation_results.csv", index=True)


if __name__ == "__main__":
    KCV()
    end_time = time.time()
    end_datetime = datetime.now()
    duration = end_time - start_time
    print("\n" + "=" * 60)
    print(f"🏁 集成训练结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {duration // 3600:.0f}小时 {(duration % 3600) // 60:.0f}分钟 {duration % 60:.1f}秒")
    print(f"📊 平均每折耗时: {duration / 5:.1f}秒")
    print("=" * 60)