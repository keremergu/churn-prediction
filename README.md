# Müşteri Ayrılma (Churn) Tahmini - Uçtan Uca Makine Öğrenmesi (Final Ödevi)

Bu proje, Makine Öğrenmesi Final Ödevi kapsamında hazırlanmıştır. Amacı, müşteri verileri üzerinden hizmetten ayrılma durumunu (churn) tahmin eden uçtan uca bir sınıflandırma problemi çözmektir.

## Veri Seti ve Problem Türü
- **Problem Türü:** İkili Sınıflandırma (Binary Classification)
- **Hedef Değişken:** `churn` (0: Kalır, 1: Ayrılır)
- **Veri Seti:** Scikit-learn kütüphanesi ve sentetik veri üretim teknikleri kullanılarak en az 200 satırlık, müşteri demografisi ve kullanım alışkanlıklarını içeren bir veri seti kurgulanmıştır.

## Kurulum ve Çalıştırma
1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt