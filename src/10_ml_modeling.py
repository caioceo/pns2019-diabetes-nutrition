"""
Script: 10_ml_modeling.py
Objetivo: Modelagem preditiva de Diabetes Mellitus Tipo 2 via Balanced Batch
          Undersampling — treina N modelos em batches balanceados 1:1 (todos os
          diabéticos + amostra aleatória de saudáveis do mesmo tamanho) e agrega
          predições por votação majoritária (ensemble).

Input:
    data/processed/09_pns2019_selected_features.csv
Output:
    docs/relatorios/resultados_modelagem.txt
    docs/relatorios/plots/ml/ (confusion matrix, ROC, feature importance)
    data/processed/10_ml_results.csv

Referências:
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    - Liu et al. (2009). Exploratory Undersampling for Class-Imbalance Learning.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_INPUT = os.path.join(BASE_DIR, "data", "processed", "09_pns2019_selected_features.csv")
DIR_PLOTS = os.path.join(BASE_DIR, "docs", "relatorios", "plots", "ml")
DIR_RELATORIOS = os.path.join(BASE_DIR, "docs", "relatorios")
PATH_RESULTS_CSV = os.path.join(BASE_DIR, "data", "processed", "10_ml_results.csv")

os.makedirs(DIR_PLOTS, exist_ok=True)
os.makedirs(DIR_RELATORIOS, exist_ok=True)

TARGET_COL = "Q03001"


def preparar_dados(df):
    """Codifica categóricas e binariza target."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('_AUSENTE_')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  [ENCODED] {col}: {list(le.classes_)}")
    
    n_nan = df.drop(columns=[TARGET_COL]).isna().sum().sum()
    if n_nan > 0:
        print(f"  [INFO] Preenchendo {n_nan} NaN residuais com -1")
        df = df.fillna(-1)
    
    # Target: 1 = diabético, 0 = saudável
    df[TARGET_COL] = df[TARGET_COL].map({1.0: 1, 2.0: 0})
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y, label_encoders


def gerar_batches(X, y, n_batches=None, seed=42):
    """
    Gera batches balanceados 1:1.
    Cada batch contém TODOS os diabéticos + uma amostra aleatória de saudáveis
    do mesmo tamanho. Os saudáveis são embaralhados e divididos sem reposição
    até esgotar, depois re-embaralhados para batches extras.
    """
    rng = np.random.RandomState(seed)
    
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)
    
    if n_batches is None:
        n_batches = n_neg // n_pos  # ~36 batches
    
    # Embaralhar saudáveis
    idx_neg_shuffled = idx_neg.copy()
    rng.shuffle(idx_neg_shuffled)
    
    batches = []
    for i in range(n_batches):
        start = (i * n_pos) % n_neg
        end = start + n_pos
        
        if end <= n_neg:
            batch_neg = idx_neg_shuffled[start:end]
        else:
            # Wrap around: pegar do final + re-embaralhar e pegar do início
            batch_neg = np.concatenate([
                idx_neg_shuffled[start:],
                idx_neg_shuffled[:end - n_neg]
            ])
            rng.shuffle(idx_neg_shuffled)  # Re-embaralhar para próximos batches
        
        batch_idx = np.concatenate([idx_pos, batch_neg])
        rng.shuffle(batch_idx)  # Embaralhar a ordem dentro do batch
        batches.append(batch_idx)
    
    return batches


def balanced_batch_ensemble(nome_modelo, criar_modelo_fn, X, y, n_batches, feature_names):
    """
    Treina N modelos em batches balanceados 1:1.
    Agrega predições por votação majoritária e probabilidades por média.
    Avalia no dataset COMPLETO (todos os 20.509 pacientes).
    """
    print(f"\n{'='*70}")
    print(f"  BALANCED BATCH ENSEMBLE: {nome_modelo}")
    print(f"  {n_batches} batches × {(y==1).sum()*2} instâncias cada (1:1)")
    print(f"{'='*70}")
    
    X_np = X.values
    y_np = y.values
    n_pos = (y == 1).sum()
    
    batches = gerar_batches(X, y, n_batches=n_batches, seed=RANDOM_STATE)
    
    # Acumuladores
    all_probas = np.zeros((len(y), n_batches))
    all_importances = np.zeros((len(feature_names), n_batches)) if nome_modelo != "Logistic Regression" else None
    batch_metrics = []
    
    for i, batch_idx in enumerate(batches):
        model = criar_modelo_fn()
        
        X_batch = X_np[batch_idx]
        y_batch = y_np[batch_idx]
        
        # Treinar no batch balanceado
        model.fit(X_batch, y_batch)
        
        # Predizer no dataset COMPLETO
        proba = model.predict_proba(X_np)[:, 1]
        all_probas[:, i] = proba
        
        # Feature importance (se disponível)
        if hasattr(model, 'feature_importances_') and all_importances is not None:
            all_importances[:, i] = model.feature_importances_
        
        # Métricas do batch (avaliando no dataset completo)
        y_pred_batch = (proba >= 0.5).astype(int)
        f1 = f1_score(y_np, y_pred_batch, zero_division=0)
        rec = recall_score(y_np, y_pred_batch, zero_division=0)
        auc = roc_auc_score(y_np, proba)
        batch_metrics.append({'batch': i+1, 'f1': f1, 'recall': rec, 'auc': auc})
        
        if (i+1) % 6 == 0 or i == 0 or i == n_batches - 1:
            print(f"  Batch {i+1:02d}/{n_batches}: F1={f1:.4f} | Recall={rec:.4f} | AUC={auc:.4f}")
    
    # Agregar: média das probabilidades
    mean_proba = all_probas.mean(axis=1)
    
    # Testar múltiplos thresholds para encontrar o melhor F1
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.30, 0.71, 0.01):
        y_pred_t = (mean_proba >= thresh).astype(int)
        f1_t = f1_score(y_np, y_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
    
    print(f"\n  [OTIMIZAÇÃO] Melhor threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    
    # Predição final com threshold otimizado
    y_pred_final = (mean_proba >= best_thresh).astype(int)
    
    # Métricas finais (ensemble agregado)
    metricas = {
        'accuracy': accuracy_score(y_np, y_pred_final),
        'precision': precision_score(y_np, y_pred_final, zero_division=0),
        'recall': recall_score(y_np, y_pred_final, zero_division=0),
        'f1': f1_score(y_np, y_pred_final, zero_division=0),
        'roc_auc': roc_auc_score(y_np, mean_proba),
        'threshold': best_thresh,
        'n_batches': n_batches
    }
    
    print(f"\n  RESULTADO ENSEMBLE ({n_batches} batches agregados):")
    print(f"  {'ACCURACY':<12}: {metricas['accuracy']:.4f}")
    print(f"  {'PRECISION':<12}: {metricas['precision']:.4f}")
    print(f"  {'RECALL':<12}: {metricas['recall']:.4f}")
    print(f"  {'F1':<12}: {metricas['f1']:.4f}")
    print(f"  {'ROC-AUC':<12}: {metricas['roc_auc']:.4f}")
    
    # Classification Report
    report = classification_report(y_np, y_pred_final, target_names=['Saudável', 'Diabético'], zero_division=0)
    print(f"\n{report}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_np, y_pred_final)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'Matriz de Confusão — {nome_modelo}\n({n_batches} Batches Balanceados, Threshold={best_thresh:.2f})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11)
    ax.set_xlabel('Classe Predita', fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Saudável', 'Diabético'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Saudável', 'Diabético'])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im)
    plt.tight_layout()
    safe = nome_modelo.replace(' ', '_')
    plt.savefig(os.path.join(DIR_PLOTS, f"cm_batch_{safe}.png"), dpi=150)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_np, mean_proba)
    auc_val = metricas['roc_auc']
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatório')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=11)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=11)
    ax.set_title(f'Curva ROC — {nome_modelo} ({n_batches} Batches)', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"roc_batch_{safe}.png"), dpi=150)
    plt.close()
    
    # Feature Importance (média dos batches)
    if all_importances is not None:
        mean_imp = all_importances.mean(axis=1)
        std_imp = all_importances.std(axis=1)
        indices = np.argsort(mean_imp)[::-1]
        
        print(f"\n  Feature Importance (média de {n_batches} batches):")
        for rank, idx in enumerate(indices, 1):
            print(f"    {rank:02d}. {feature_names[idx]:<35} {mean_imp[idx]:.4f} ± {std_imp[idx]:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        top_n = len(feature_names)
        top_idx = indices[:top_n]
        ax.barh(range(top_n), mean_imp[top_idx], xerr=std_imp[top_idx],
                color='#4CAF50', edgecolor='white', capsize=3)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
        ax.set_xlabel('Importância Gini (média ± DP)', fontsize=11)
        ax.set_title(f'Feature Importance — {nome_modelo} ({n_batches} Batches)', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_PLOTS, f"importance_batch_{safe}.png"), dpi=150)
        plt.close()
        
        metricas['top_features'] = [(feature_names[idx], mean_imp[idx]) for idx in indices]
    
    metricas['report'] = report
    metricas['cm'] = cm
    
    return metricas


def main():
    print("=" * 70)
    print("  MODELAGEM PREDITIVA — BALANCED BATCH UNDERSAMPLING | PNS 2019")
    print("=" * 70)
    
    df = pd.read_csv(PATH_INPUT)
    n_pos = (df[TARGET_COL] == 1).sum()
    n_neg = (df[TARGET_COL] == 2).sum()
    n_batches = n_neg // n_pos  # 19954 // 555 = 35
    
    print(f"\n[INFO] Base: {df.shape[0]:,} registros × {df.shape[1]} colunas")
    print(f"[INFO] Diabéticos: {n_pos} | Saudáveis: {n_neg}")
    print(f"[INFO] Batches possíveis: {n_batches} (cada um com {n_pos} diabéticos + {n_pos} saudáveis = {n_pos*2})")
    
    print("\n[INFO] Preparando dados...")
    X, y, _ = preparar_dados(df)
    feature_names = list(X.columns)
    
    # ── Modelo 1: Logistic Regression com Batches ──
    def criar_lr():
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
    
    metricas_lr = balanced_batch_ensemble(
        "Logistic Regression", criar_lr, X, y, n_batches, feature_names
    )
    
    # ── Modelo 2: Random Forest com Batches ──
    def criar_rf():
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=None, n_jobs=-1  # None para diversidade entre batches
        )
    
    metricas_rf = balanced_batch_ensemble(
        "Random Forest", criar_rf, X, y, n_batches, feature_names
    )
    
    # ── Salvar Resultados ──
    resultados = {
        "Logistic Regression (Batch Ensemble)": {k: v for k, v in metricas_lr.items() if k not in ['report', 'cm', 'top_features']},
        "Random Forest (Batch Ensemble)": {k: v for k, v in metricas_rf.items() if k not in ['report', 'cm', 'top_features']},
    }
    df_res = pd.DataFrame(resultados).T
    df_res.index.name = "Modelo"
    df_res.to_csv(PATH_RESULTS_CSV, encoding="utf-8")
    print(f"\n[SALVO] {PATH_RESULTS_CSV}")
    
    # ── Relatório Textual ──
    report_path = os.path.join(DIR_RELATORIOS, "resultados_modelagem.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RELATÓRIO — BALANCED BATCH UNDERSAMPLING | PNS 2019\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Estratégia: {n_batches} batches balanceados 1:1\n")
        f.write(f"Cada batch: {n_pos} diabéticos (fixos) + {n_pos} saudáveis (aleatórios)\n")
        f.write(f"Agregação: Média das probabilidades + threshold otimizado por F1\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'Modelo':<40} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Thresh':>8}\n")
        f.write("-" * 80 + "\n")
        for nome, m in resultados.items():
            f.write(f"{nome:<40} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f} {m['threshold']:>8.2f}\n")
        f.write("-" * 80 + "\n\n")
        
        for nome, met in [("Logistic Regression", metricas_lr), ("Random Forest", metricas_rf)]:
            f.write(f"\n--- {nome} ---\n")
            f.write(met['report'])
            f.write(f"\nConfusion Matrix:\n{met['cm']}\n")
            if 'top_features' in met:
                f.write(f"\nFeature Importance (top 19):\n")
                for rank, (feat, imp) in enumerate(met['top_features'], 1):
                    f.write(f"  {rank:02d}. {feat:<35} {imp:.4f}\n")
    
    print(f"[SALVO] {report_path}")
    print(f"[SALVO] Plots em: {DIR_PLOTS}/")
    
    # ── Resumo Final ──
    best = max(resultados.items(), key=lambda x: x[1]['f1'])
    print(f"\n{'='*70}")
    print(f"  MELHOR: {best[0]}")
    print(f"  F1={best[1]['f1']:.4f} | AUC={best[1]['roc_auc']:.4f} | Recall={best[1]['recall']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
