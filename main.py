"""
Ana Çalıştırma Scripti (Pipeline Runner)
"""
import sys
import os

# Kök dizini modül arama yoluna ekle
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data, check_missing_values
from src.preprocessor import analyze_and_plot_outliers, encode_categorical_features, split_and_scale_data
from src.train import get_models, train_and_evaluate_all
from src.evaluate import (
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_metrics_comparison
)

def run_pipeline():
    print("="*60)
    print("KALP HASTALIGI TAHMINI - UCTAN UCA MAKINE OGRENMESI PIPELINE")
    print("="*60)
    
    # 1. Veri Yükleme
    df = load_data()
    check_missing_values(df)
    
    # 2. Aykırı Değer Analizi
    analyze_and_plot_outliers(df)
    
    # 3. Encoding
    df_encoded = encode_categorical_features(df)
    
    # 4. Bölme ve Ölçekleme
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = split_and_scale_data(df_encoded)
    
    # 5. Model Eğitimi
    models = get_models(random_state=42)
    results, metrics_df = train_and_evaluate_all(models, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 6. Görselleştirmelerin Üretilmesi
    print("\n" + "="*60)
    print("GÖRSELLEŞTİRMELER VE RAPORLAR OLUŞTURULUYOR")
    print("="*60)
    plot_metrics_comparison(metrics_df)
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_feature_importance(results['Gradient Boosting']['model'], feature_names)
    
    print("\n[BASARILI] Pipeline basariyla tamamlandi! Tum grafikler 'reports/figures/' dizinine kaydedildi.")

if __name__ == "__main__":
    run_pipeline()
