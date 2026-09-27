"""
Veri Yükleme ve Ön Kontrol Modülü
"""
import os
import pandas as pd

DEFAULT_URL = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

def load_data(url: str = DEFAULT_URL, cache_path: str = "data/raw/heart.csv") -> pd.DataFrame:
    """
    Veri setini internetten otomatik çeker veya yerel önbellekten yükler.
    """
    if os.path.exists(cache_path):
        print(f"[BILGI] Yerel veri kopyasi yukleniyor: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        print(f"[BILGI] Veri seti internetten cekiliyor: {url}")
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[OK] Veri seti yerel dizine kaydedildi: {cache_path}")
    
    print(f"[OK] Veri boyutu: {df.shape[0]} satir, {df.shape[1]} sutun.")
    return df

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Eksik veri kontrolü yapar ve raporlar.
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[OK] Veri setinde eksik (NaN) deger bulunmamaktadir.")
    else:
        print(f"[UYARI] Toplam {missing.sum()} adet eksik deger tespit edildi.")
    return missing
