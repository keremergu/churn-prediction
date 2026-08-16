"""
Logistic Regression Modeli (Uçtan Uca Eğitim ve Test)
Ödev Maddeleri: 12, 13, 14, 15, 16, 17
"""

from veri_isleme import (
    X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train, y_val, y_test, secilen_oznitelikler
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

print("\n" + "="*60)
print("1. LOJİSTİK REGRESYON MODELİ (EĞİTİM VE OPTİMİZASYON)")
print("="*60)

# Madde 12 & 14: GridSearchCV ile hiperparametre arama ve model eğitimi
print("Model farklı parametrelerle eğitiliyor ve en iyisi aranıyor...\n")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1'
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

# Madde 17: Katsayıları (Özelliklerin Etkisini) Yazdırma
katsayilar = pd.DataFrame({
    'Öznitelik': secilen_oznitelikler,
    'Katsayı (Etki)': best_model.coef_[0]
}).sort_values(by='Katsayı (Etki)', key=abs, ascending=False)

print(katsayilar.to_string(index=False))

# Madde 16: Model Yorumu
print("\n--- Model Yorumu ---")
print("Lojistik Regresyon modeli arka planda kurduğu doğrusal denklem sayesinde")
print("müşterinin ayrılma (churn) kararına hangi değişkenin ne kadar etki ettiğini")
print("açıkça göstermektedir. Yukarıdaki tabloda katsayısı en yüksek ve pozitif olan")
print("özellikler müşteriyi ayrılmaya iterken, negatif olanlar sistemde tutmaktadır.")
print("GridSearchCV kullanılarak modelin aşırı öğrenmesi (overfitting) engellenmiştir.")
print("="*60)