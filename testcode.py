import os
import re
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from models.DLS import classifier
from utils.one_trial import LitModel
from utils.prepare_data import get_Ksucc

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    confusion_matrix,
    recall_score,
    average_precision_score,
)

def load_config(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collect_ckpts(ckpt_dir: str, name_hint: str | None = None):
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"ckpt 目录不存在: {ckpt_dir}")
    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if name_hint:
        filtered = [p for p in ckpts if name_hint in os.path.basename(p)]
        if filtered:
            ckpts = filtered
    if not ckpts:
        raise FileNotFoundError(f"未在 {ckpt_dir} 找到任何 .ckpt 文件")
    return ckpts

def _select_ckpts(
    ckpt_files: list[str],
    strategy: str = "latest",
    count: int = 5,
    monitor: str | None = None,
    mode: str = "max",
):

    if not ckpt_files:
        return []
    strategy = strategy.lower()
    if strategy == "latest":
        return sorted(ckpt_files, key=lambda p: os.path.getmtime(p), reverse=True)[:count]

    if strategy == "best":
        scored = []
        for p in ckpt_files:
            fname = os.path.basename(p)
            # Try to extract the last floating number, which in PL filenames is usually the monitored metric
            m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", fname)
            score = None
            if m:
                try:
                    score = float(m[-1])
                except Exception:
                    score = None
            if score is not None:
                scored.append((p, score))
        if not scored:
            # Fallback to latest if no scores can be parsed
            return sorted(ckpt_files, key=lambda p: os.path.getmtime(p), reverse=True)[:count]
        reverse = True if mode == "max" else False
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=reverse)
        return [p for p, _ in scored_sorted[:count]]

    # default fallback
    return sorted(ckpt_files, key=lambda p: os.path.getmtime(p), reverse=True)[:count]

def infer_softvote_from_ckpts(
    ckpt_dir: str = "ckpt",
    cfg_path: str = "config.yaml",
    strategy: str = "latest",
    count: int = 5,
):
    # 时间日志
    start_time = datetime.now()
    print("=" * 60)
    print(f"🧪 推理开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 配置与数据
    config = load_config(cfg_path)
    hparams = classifier.get_hparams()
    model_params = classifier.get_model_params()

    # 数据集
    trainset, testset = get_Ksucc()
    test_labels = [testset[i][1] for i in range(len(testset))]

    # 输出目录
    output_dir = f"result/{config['name']}result"
    os.makedirs(output_dir, exist_ok=True)

    # 权重文件
    all_ckpts = collect_ckpts(ckpt_dir, name_hint=config.get('name'))
    monitor_name = config.get('monitor', 'valid_S')
    mode = config.get('mode', 'max')
    ckpt_files = _select_ckpts(all_ckpts, strategy=strategy, count=count, monitor=monitor_name, mode=mode)
    print(f"找到 {len(all_ckpts)} 个候选权重，按 {strategy} 选择 {len(ckpt_files)} 个用于 Soft Voting：")
    for p in ckpt_files:
        print(" -", os.path.basename(p))

    # 逐权重推理
    test_proba_list = []
    for ckpt_path in ckpt_files:
        model = LitModel(classifier, hparams, model_params)
        model.set_ckptdir(ckpt_dir)
        proba_tensor = model.predict_proba(testset, ckpt_path=ckpt_path)
        proba = proba_tensor.cpu().numpy().flatten()
        test_proba_list.append(proba)

    avg_proba = np.mean(test_proba_list, axis=0)
    soft_pred = (avg_proba >= 0.5).astype(int)

    # 指标
    acc = accuracy_score(test_labels, soft_pred)
    f1 = f1_score(test_labels, soft_pred)
    mcc = matthews_corrcoef(test_labels, soft_pred)
    sn = recall_score(test_labels, soft_pred)
    #tn, fp, fn, tp = confusion_matrix(test_labels, soft_pred).ravel()
    #sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    #bacc = (sn + sp) / 2

    auc_score = roc_auc_score(test_labels, avg_proba)
    precision, recall, _ = precision_recall_curve(test_labels, avg_proba)
    aupr = auc(recall, precision)
    ap_score = average_precision_score(test_labels, avg_proba)

    # 打印
    print("\nSoftVoting(ckpts) 评估结果：")
    #print(f"Recall(SN): {sn:.4f}")
    #print(f"Specificity(SP): {sp:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"AUC       : {auc_score:.4f}")
    print(f"AP        : {ap_score:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"AUPR      : {aupr:.4f}")
    #print(f"bacc      : {bacc:.4f}")

    # 保存
    test_df = pd.read_csv("data/suc/test_data.csv").copy()

    test_df["probability"] = avg_proba
    test_df["predicted_label"] = soft_pred

    test_df.to_csv(
    f"{output_dir}/soft_from_ckpts_ensemble_result.csv",
    index=False
)

    top10_df = test_df.sort_values("probability", ascending=False).head(10)
    top10_df.to_csv(
    f"{output_dir}/top10_high_confidence_sites.csv",
    index=False
)
    
    """
    pd.DataFrame([
        {
            'SN': sn, 'SP': sp, 'Accuracy': acc, 'AUC': auc_score,
            'AP': ap_score, 'MCC': mcc, 'F1': f1, 'AUPR': aupr, 'bacc': bacc
        }
    ]).to_csv(f"{output_dir}/soft_from_ckpts_metrics.csv", index=False)"""

    end_time = datetime.now()
    print("-" * 60)
    print(f"🏁 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  耗时: {(end_time - start_time).total_seconds():.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    infer_softvote_from_ckpts(
        ckpt_dir="ckpt/dls-suc",
        cfg_path="config.yaml",
        strategy="latest",
        count=5,
    )
