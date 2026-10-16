"""
Model Tanımlama, Eğitim ve Metrik Değerlendirme Modülü
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

from src.preprocessor import CONTINUOUS_FEATURES


def get_models(random_state: int = 42) -> dict:
    """
    Karşılaştırılan 3 algoritmayı yapılandırır.
    """
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=3, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_split=4, random_state=random_state
        ),
        "Gaussian Naive Bayes": GaussianNB()
    }


def train_and_evaluate_all(models: dict, X_train, X_test, y_train, y_test) -> tuple:
    """
    Modelleri eğitir ve test setindeki metrikleri hesaplar (pozitif sınıf = hasta).
    """
    results = {}
    print("="*60)
    print("MODELLER EGITILIYOR VE PERFORMANSLAR HESAPLANIYOR")
    print("="*60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'cm': cm
        }
        print(f"[OK] {name:22s} | Dogruluk: %{acc*100:.2f} | AUC: {roc_auc:.4f} | Recall: %{rec*100:.2f}")

    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results],
        'ROC-AUC': [results[m]['roc_auc'] for m in results]
    }).set_index('Model')

    return results, metrics_df


def cross_validate_models(models: dict, X, y, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Tek bir train/test bölmesi küçük veri setinde (61 test örneği) şansa bağlı
    sonuç verebileceğinden, modeller tabakalı 5 katlı çapraz doğrulamayla da ölçülür.
    Ölçekleyici her katta yalnızca o katın eğitim kısmına fit edilir (Pipeline).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {'Accuracy': 'accuracy', 'Recall': 'recall', 'F1-Score': 'f1', 'ROC-AUC': 'roc_auc'}
    rows = {}
    print("="*60)
    print(f"{n_splits} KATLI CAPRAZ DOGRULAMA (ortalama +- standart sapma)")
    print("="*60)
    for name, model in models.items():
        pipe = Pipeline([
            ('scale', ColumnTransformer([('num', StandardScaler(), CONTINUOUS_FEATURES)],
                                        remainder='passthrough')),
            ('model', model),
        ])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        rows[name] = {m: f"{scores['test_' + m].mean():.3f} ± {scores['test_' + m].std():.3f}" for m in scoring}
        print(f"[CV] {name:22s} | " + " | ".join(f"{m}: {v}" for m, v in rows[name].items()))
    return pd.DataFrame(rows).T
