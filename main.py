"""
Ana Çalıştırma Scripti (Pipeline Runner)
"""
import sys
import os

# Göreli yollar (data/, reports/) script nereden çalıştırılırsa çalıştırılsın proje köküne göre olsun
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)
os.chdir(ROOT)

from src.data_loader import load_data, check_missing_values
from src.preprocessor import (
    fix_target_labels, analyze_and_plot_outliers, encode_categorical_features, split_and_scale_data
)
from src.train import get_models, train_and_evaluate_all, cross_validate_models
from src.evaluate import (
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_metrics_comparison
)


def run_pipeline():
    print("="*60)
    print("KALP HASTALIGI TAHMINI - UCTAN UCA MAKINE OGRENMESI PIPELINE")
    print("="*60)

    # 1. Veri Yükleme ve hedef etiketin klinik anlama çevrilmesi
    df = load_data()
    check_missing_values(df)
    df = fix_target_labels(df)

    # 2. Aykırı Değer Analizi
    analyze_and_plot_outliers(df)

    # 3. Encoding
    df_encoded = encode_categorical_features(df)

    # 4. Bölme ve Ölçekleme
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = split_and_scale_data(df_encoded)

    # 5. Model Eğitimi (test seti) ve 5 katlı çapraz doğrulama
    models = get_models(random_state=42)
    results, metrics_df = train_and_evaluate_all(models, X_train_scaled, X_test_scaled, y_train, y_test)
    cross_validate_models(get_models(random_state=42),
                          df_encoded.drop('target', axis=1), df_encoded['target'])

    # 6. Görselleştirmelerin Üretilmesi
    print("\n" + "="*60)
    print("GORSELLESTIRMELER VE RAPORLAR OLUSTURULUYOR")
    print("="*60)
    plot_metrics_comparison(metrics_df)
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_feature_importance(results['Gradient Boosting']['model'], feature_names)

    print("\n[BASARILI] Pipeline basariyla tamamlandi! Tum grafikler 'reports/figures/' dizinine kaydedildi.")


if __name__ == "__main__":
    run_pipeline()
