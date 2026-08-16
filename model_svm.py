"""
Support Vector Machine (SVM) Modeli (Uçtan Uca Eğitim ve Test)
Ödev Maddeleri: 12, 13, 14, 15, 16, 17
"""

from veri_isleme import (
    X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train, y_val, y_test, secilen_oznitelikler
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import numpy as np

print("\n" + "="*60)
print("1. SVM MODELİ (EĞİTİM VE OPTİMİZASYON)")
print("="*60)

# Madde 12 & 14: GridSearchCV ile hiperparametre arama
print("Farklı C değerleri ve Kernel (Çekirdek) tipleri deneniyor...\n")
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
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

# Madde 17: SVM için Açıklanabilirlik
if best_model.kernel == 'linear':
    # Sadece doğrusal (linear) kernel kullanıldığında katsayılar alınabilir
    katsayilar = pd.DataFrame({
        'Öznitelik': secilen_oznitelikler,
        'Katsayı (Etki)': best_model.coef_[0]
    }).sort_values(by='Katsayı (Etki)', key=abs, ascending=False)
    print(katsayilar.to_string(index=False))
else:
    print(f"Seçilen en iyi kernel '{best_model.kernel}' olduğu için (doğrusal değil),")
    print("SVM doğrudan katsayı (feature importance) üretemez. Bu durum RBF kernelin")
    print("boyutlar arası karmaşık (non-linear) dönüşümler yapmasından kaynaklanır.")

# Madde 16: Model Yorumu
print("\n--- Model Yorumu ---")
print("Support Vector Machine (SVM) modeli, sınıflar arasındaki marjı (boşluğu)")
print("maksimize eden optimum hiperdüzlemi bulmaya çalışır. GridSearchCV kullanılarak")
print("hem doğrusal (linear) hem de doğrusal olmayan (rbf) çekirdekler test edilmiş,")
print("en yüksek F1 skorunu veren parametreler test seti üzerinde değerlendirilmiştir.")
print("="*60)