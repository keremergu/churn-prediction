"""
Veri Ön İşleme ve Hazırlık Modülü
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("1. VERİ SETİ AÇIKLAMASI VE PROBLEM TÜRÜ")
print("=" * 60)
print("Problem Türü: İkili Sınıflandırma (Binary Classification - Müşteri Ayrılma Tahmini)")
print("Hedef Değişken: 'churn' (0 = Kalır, 1 = Ayrılır)\n")

# 1. VERİ ÜRETİMİ
np.random.seed(42)
n_samples = 500
yas = np.random.randint(18, 70, size=n_samples)
gelir = np.random.normal(loc=60000, scale=20000, size=n_samples) 
abonelik_suresi = np.random.randint(1, 72, size=n_samples)
destek_talebi_sayisi = np.random.poisson(lam=2, size=n_samples)
sehir = np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Bursa'], size=n_samples)

log_odds = (0.04 * (yas - 40) - 0.00002 * (gelir - 60000) - 0.05 * (abonelik_suresi - 24) + 0.6 * (destek_talebi_sayisi - 2))
churn_probs = 1 / (1 + np.exp(-log_odds))
churn = (churn_probs > 0.5).astype(int)

df = pd.DataFrame({'yas': yas, 'gelir': gelir, 'abonelik_suresi': abonelik_suresi, 'destek_talebi_sayisi': destek_talebi_sayisi, 'sehir': sehir, 'churn': churn})

print("=" * 60)
print("2. TEMEL VERİ İNCELEME ÇIKTILARI (EDA)")
print("=" * 60)
print(f"Oluşturulan Veri Seti Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun\n")
print("İlk 5 Satır:")
print(df.head(), "\n")
print("Sayısal Değişken İstatistikleri:")
print(df.describe().round(2), "\n")

# 2. EKSİK VE AYKIRI DEĞER YÖNETİMİ
print("=" * 60)
print("3. VERİ ÖN İŞLEME ADIMLARI")
print("=" * 60)
eksik_degerler = df.isnull().sum().sum()
print(f"Toplam Eksik Değer Sayısı: {eksik_degerler}")
if eksik_degerler > 0:
    df['yas'].fillna(df['yas'].median(), inplace=True)
    df['sehir'].fillna(df['sehir'].mode()[0], inplace=True)
    print("-> Eksik değerler medyan/mod ile dolduruldu.")

Q1 = df['gelir'].quantile(0.25)
Q3 = df['gelir'].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

aykiri_sayisi = df[(df['gelir'] < alt_sinir) | (df['gelir'] > ust_sinir)].shape[0]
print(f"Gelir sütununda tespit edilen aykırı değer sayısı: {aykiri_sayisi}")

df['gelir'] = np.where(df['gelir'] < alt_sinir, alt_sinir, df['gelir'])
df['gelir'] = np.where(df['gelir'] > ust_sinir, ust_sinir, df['gelir'])
print("-> Aykırı değerler IQR yöntemiyle alt ve üst sınırlara baskılandı (Capping).\n")

# 3. FEATURE ENGINEERING & ENCODING
print("=" * 60)
print("4. ÖZNİTELİK MÜHENDİSLİĞİ VE SEÇİMİ")
print("=" * 60)
df['gelir_grubu'] = pd.qcut(df['gelir'], q=3, labels=['Dusuk', 'Orta', 'Yuksek'])
df['destek_talebi_var_mi'] = (df['destek_talebi_sayisi'] > 0).astype(int)
print("-> 'gelir_grubu' ve 'destek_talebi_var_mi' adında 2 yeni öznitelik türetildi.")

df_encoded = pd.get_dummies(df, columns=['sehir', 'gelir_grubu'], drop_first=True)
print("-> Kategorik değişkenlere One-Hot Encoding uygulandı.")

# 4. FEATURE SELECTION
X = df_encoded.drop(columns=['churn'])
y = df_encoded['churn']

korelasyonlar = X.corrwith(y).abs()
secilen_oznitelikler = korelasyonlar[korelasyonlar > 0.01].index
X_secilen = X[secilen_oznitelikler]
print(f"-> Hedef değişkenle korelasyonu düşük olanlar elendi. Seçilen öznitelik sayısı: {len(secilen_oznitelikler)}")

# 5. TRAIN / VAL / TEST BÖLME VE ÖLÇEKLEME
X_train_val, X_test, y_train_val, y_test = train_test_split(X_secilen, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15/0.85), stratify=y_train_val, random_state=42)

scaler = StandardScaler()
sayisal_sutunlar = ['yas', 'gelir', 'abonelik_suresi', 'destek_talebi_sayisi']
sayisal_sutunlar_secilen = [col for col in sayisal_sutunlar if col in X_train.columns]

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[sayisal_sutunlar_secilen] = scaler.fit_transform(X_train[sayisal_sutunlar_secilen])
X_val_scaled[sayisal_sutunlar_secilen] = scaler.transform(X_val[sayisal_sutunlar_secilen])
X_test_scaled[sayisal_sutunlar_secilen] = scaler.transform(X_test[sayisal_sutunlar_secilen])

print("-> Veri; Train, Validation ve Test olarak başarıyla bölündü ve StandardScaler ile ölçeklendi.")
print("=" * 60)