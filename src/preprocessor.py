"""
Veri Ön İşleme, Encoding, Outlier ve Ölçekleme Modülü
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'ca', 'thal']

def analyze_and_plot_outliers(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Kutu grafiği ile aykırı değerleri inceler ve kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    for i, col in enumerate(CONTINUOUS_FEATURES):
        sns.boxplot(y=df[col], ax=axes[i], color='#4c72b0', flierprops={'markerfacecolor':'#c44e52', 'markersize': 6})
        axes[i].set_title(f"{col}", fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Ölçüm' if i == 0 else '')
    
    plt.suptitle("Sürekli Değişkenlerde Aykırı Değer (Outlier) Analizi", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "outlier_boxplots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] Aykiri deger analizi kaydedildi: {plot_path}")

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorik özellikleri One-Hot Encoding ile dönüştürür.
    """
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)
    print(f"[OK] One-Hot Encoding tamamlandi. Sutun sayisi: {df.shape[1]} -> {df_encoded.shape[1]}")
    return df_encoded

def split_and_scale_data(df_encoded: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Veriyi %80 train - %20 test olarak böler ve StandardScaler ile ölçekler.
    Ölçekleme öncesi/sonrası dağılım grafiğini kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_raw = X_train[CONTINUOUS_FEATURES].copy()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])
    
    # Dağılım Grafiği (Önce vs Sonra)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for idx, col in enumerate(CONTINUOUS_FEATURES):
        sns.kdeplot(X_train_raw[col], ax=axes[0, idx], color='#2b5c8f', fill=True, alpha=0.35, lw=2)
        axes[0, idx].set_title(f"Önce: {col}\nOrt: {X_train_raw[col].mean():.1f}, Std: {X_train_raw[col].std():.1f}", fontsize=10, fontweight='bold')
        axes[0, idx].set_xlabel('')
        axes[0, idx].set_ylabel('Yoğunluk' if idx == 0 else '')
        
        sns.kdeplot(X_train_scaled[col], ax=axes[1, idx], color='#2ca02c', fill=True, alpha=0.35, lw=2)
        axes[1, idx].set_title(f"Sonra: {col} (Scaled)\nOrt: ~0.0, Std: 1.0", fontsize=10, fontweight='bold')
        axes[1, idx].set_xlabel('Ölçekli Değer')
        axes[1, idx].set_ylabel('Yoğunluk' if idx == 0 else '')
        
    plt.suptitle("Sürekli Özelliklerin Ölçekleme Öncesi ve Sonrası Dağılım Karşılaştırması", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "scaling_distribution_comparison.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] Olcekleme dagilim karsilastirmasi kaydedildi: {dist_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
