# Müşteri Ayrılma (Churn) Tahmini - Makine Öğrenmesi Akışı

Bu proje, müşteri verileri üzerinden müşteri kaybını (churn) tahmin etmek amacıyla temel makine öğrenmesi akışını (veri oluşturma, ön işleme, özellik mühendisliği, model eğitimi ve değerlendirme) tek bir Python dosyasında uygular.

## Projenin Amacı
Müşterilerin demografik özellikleri ve kullanım alışkanlıklarını kullanarak hizmetten ayrılıp ayrılmayacağını (churn) tahmin eden bir sınıflandırma modeli geliştirmektir.

## Kullanılan Kütüphaneler
- `pandas`
- `numpy`
- `scikit-learn`

## Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Projeyi çalıştırın:
```bash
python main.py
```

## Yapılan İşlemler
1. **Veri Üretimi & İnceleme:** 300 satırlık sentetik müşteri verisi üretildi ve temel istatistikleri incelendi.
2. **Ön İşleme:** Eksik değer kontrolü yapıldı, kategorik değişkenlere One-Hot Encoding uygulandı ve sayısal veriler `StandardScaler` ile ölçeklendi.
3. **Özellik Mühendisliği:** Destek talebi varlığı ve gelir grubu gibi yeni öznitelikler türetildi.
4. **Modelleme:** Veri %70 Train, %15 Validation ve %15 Test olarak bölündü. Logistic Regression, KNN ve Decision Tree modelleri eğitildi.
5. **Değerlendirme:** Validation performansı en iyi olan model seçilerek Test seti üzerinde Confusion Matrix, Accuracy, Precision, Recall ve F1-Score metrikleriyle nihai değerlendirme yapıldı.