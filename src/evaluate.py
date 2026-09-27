"""
Görselleştirme ve Grafik Üretim Modülü
(Confusion Matrix, ROC Curve, Feature Importance, Metrics Comparison)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

def plot_confusion_matrices(results: dict, output_path: str = "reports/figures/confusion_matrices.png"):
    """
    Her model için TP, FP, TN, FN etiketli detaylı Confusion Matrix Heatmap çizdirir.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cm_labels = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
    
    for idx, (name, res) in enumerate(results.items()):
        cm = res['cm']
        counts = [f"{v:d}" for v in cm.flatten()]
        percentages = [f"{v:.1%}" for v in cm.flatten() / np.sum(cm)]
        annot_labels = [f"{lbl}\n\nAdet: {cnt}\n({pct})" for lbl, cnt, pct in zip(cm_labels, counts, percentages)]
        annot_matrix = np.asarray(annot_labels).reshape(2, 2)
        
        sns.heatmap(
            cm, annot=annot_matrix, fmt='', cmap='Blues', ax=axes[idx],
            cbar=False, annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            xticklabels=['Sağlıklı (0)', 'Kalp Hastası (1)'],
            yticklabels=['Sağlıklı (0)', 'Kalp Hastası (1)'],
            linewidths=2, linecolor='white'
        )
        axes[idx].set_title(f"{name}\nDoğruluk: %{res['accuracy']*100:.1f} | Recall: %{res['recall']*100:.1f}", 
                            fontsize=12, fontweight='bold', pad=12)
        axes[idx].set_xlabel('Tahmin Edilen Sınıf', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Gerçek Sınıf', fontsize=11, fontweight='bold')
        
    plt.suptitle("Modellerin Karmaşıklık Matrisleri (Confusion Matrix) — TP, FP, TN, FN Değerleri", 
                 fontsize=15, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] Confusion Matrix kaydedildi: {output_path}")

def plot_roc_curves(results: dict, y_test, output_path: str = "reports/figures/roc_curves_comparison.png"):
    """
    Üç modelin ROC eğrilerini ve AUC değerlerini tek grafikte karşılaştırır.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 7))
    model_colors = {
        'Gradient Boosting': '#1f77b4',
        'Decision Tree': '#ff7f0e',
        'Bayesian Network (GaussianNB)': '#2ca02c'
    }
    
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        auc_score = res['roc_auc']
        plt.plot(fpr, tpr, color=model_colors[name], lw=2.8, label=f"{name} (AUC = {auc_score:.3f})")
        
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.8, linestyle='--', label='Rastgele Sınıflandırıcı (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Özgüllük / FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Duyarlılık / Recall / TPR)', fontsize=12, fontweight='bold')
    plt.title('Modellerin ROC Eğrileri ve AUC Skorları Karşılaştırması', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] ROC-AUC egrisi kaydedildi: {output_path}")

def plot_feature_importance(gb_model, feature_names, output_path: str = "reports/figures/feature_importance_gradient_boosting.png"):
    """
    Gradient Boosting modeli için özellik önemi grafiğini çizer.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importances = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    bars = importances.plot(kind='barh', color='#2b5c8f', edgecolor='black', alpha=0.85, width=0.7)
    plt.title('Gradient Boosting — Özellik Önem Düzeyleri (Feature Importance)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Göreceli Önem Skoru', fontsize=12, fontweight='bold')
    plt.ylabel('Klinik ve Demografik Özellikler', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for p in bars.patches:
        val = p.get_width()
        bars.annotate(f"{val:.3f}",
                      (val, p.get_y() + p.get_height() / 2.),
                      va='center', xytext=(5, 0), textcoords='offset points',
                      fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] Feature Importance kaydedildi: {output_path}")

def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: str = "reports/figures/models_performance_comparison.png"):
    """
    Modellerin tüm metriklerini yan yana çubuk grafikle karşılaştırır.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ax = metrics_df.plot(kind='bar', figsize=(13, 6), colormap='viridis', width=0.75, edgecolor='black', alpha=0.9)
    plt.title('Modellerin Performans Metrikleri Karşılaştırması', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Skor (0 - 1.0)', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylim(0.5, 1.05)
    plt.xticks(rotation=0, fontsize=11, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3),
                        textcoords='offset points')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] Metrik karsilastirmasi kaydedildi: {output_path}")
