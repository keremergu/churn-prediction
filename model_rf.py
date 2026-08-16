"""
Random Forest Modeli (Uçtan Uca Eğitim ve Test)
Ödev Maddeleri: 12, 13, 14, 15, 16, 17
"""

from veri_isleme import (
    X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train, y_val, y_test, secilen_oznitelikler
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

print("\n" + "="*60)
print("1. RANDOM FOREST MODELİ (EĞİTİM VE OPTİMİZASYON)")
print("="*60)

# Madde 12 & 14: GridSearchCV ile hiperparametre arama
print("Farklı ağaç sayıları ve derinlikler deneniyor...\n")
param_grid = {
    'n_estimators': [50, 100, 200],     # Ormandaki ağaç sayısı
    'max_depth': [None, 5, 10, 15],     # Ağaçların maksimum derinliği
    'min_samples_split': [2, 5]         # Dallanma için gereken minimum örnek
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1 # İşlemi hızlandırmak için tüm işlemci çekirdeklerini kullan
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"-> Bulunan En İyi Parametreler: {grid_search.best_params_}")

# Madde 13: Validation seti skoru
val_preds = best_model.predict(X_val_scaled)
print(f"-> Validation (Doğrulama) F1-Score: {f1_score(y_val, val_preds):.4f}\n")

print("="*60)
print("2. TEST SETİ DEĞERLENDİRMESİ (FINAL SINAVI)")
print("="*60)

# Madde 15: Test verisi üzerinde tahmin
test_preds = best_model.predict(X_test_scaled)

print("Confusion Matrix (Karmaşıklık Matrisi):")
print(confusion_matrix(y_test, test_preds))

print("\nDetaylı Sınıflandırma Raporu (Precision, Recall, F1-Score):")
print(classification_report(y_test, test_preds, zero_division=0))

print("="*60)
print("3. AÇIKLANABİLİRLİK VE YORUM (FEATURE IMPORTANCE)")
print("="*60)

# Madde 17: Random Forest'a özel 'feature_importances_' kullanımı
onem_dereceleri = pd.DataFrame({
    'Öznitelik': secilen_oznitelikler,
    'Önem Derecesi (%)': best_model.feature_importances_ * 100
}).sort_values(by='Önem Derecesi (%)', ascending=False)

print(onem_dereceleri.round(2).to_string(index=False))

# Madde 16: Model Yorumu
print("\n--- Model Yorumu ---")
print("Random Forest modeli, birden fazla karar ağacının tahminlerini birleştirerek")
print("(Ensemble Learning) çalışır. Lojistik Regresyon'dan farklı olarak katsayı (negatif/pozitif)")
print("yerine, bir değişkenin modeli bölmede ne kadar faydalı olduğuna (Önem Derecesi) bakar.")
print("GridSearchCV ile optimum ağaç sayısı ve derinlik bulunarak modelin performansı maksimize edilmiştir.")
print("="*60)