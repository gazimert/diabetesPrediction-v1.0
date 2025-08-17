# 🩺 Diyabet Tahmin Modeli (MLP + Flask API)

Bu proje, **MLP (Multi-Layer Perceptron)** algoritması ile diyabet
tahmini yapar ve eğitilen modeli bir **Flask tabanlı REST API**
üzerinden sunar.

## 🔹 Kullanılan Teknolojiler

-   Python\
-   Pandas, NumPy (Veri işleme)\
-   Scikit-learn (Önişleme, scaler)\
-   TensorFlow / Keras (MLP modeli)\
-   Flask (REST API)\
-   Joblib (Scaler kaydetme / yükleme)

------------------------------------------------------------------------

## 🌐 Flask API Kullanımı

API'yi başlatmak için:

``` bash
python app.py
```

API varsayılan olarak `http://0.0.0.0:5001` üzerinde çalışır.

------------------------------------------------------------------------

## 📡 Tahmin Yapma (API Kullanımı)

### Endpoint

**`POST /predict`**

### Örnek JSON Girdi:

``` json
{
  "pregnancies": 2,
  "glucose": 120,
  "bloodPressure": 70,
  "skinThickness": 20,
  "insulin": 85,
  "bmi": 28.5,
  "diabetesPedigreeFunction": 0.5,
  "age": 30
}
```

### Örnek Çıktı:

``` json
{
  "prediction": 0,
  "probability": 0.34
}
```

-   `prediction`: 0 (sağlıklı) veya 1 (diyabet)\
-   `probability`: Tahmin olasılığı (0.0--1.0)

------------------------------------------------------------------------

## 📊 Model Detayları

-   **Giriş Katmanı:** 8 özellik\
-   **1. Gizli Katman:** 16 nöron (ReLU)\
-   **2. Gizli Katman:** 8 nöron (ReLU)\
-   **Çıkış Katmanı:** 1 nöron (Sigmoid)\
-   **Kayıp Fonksiyonu:** Binary Crossentropy\
-   **Optimizatör:** Adam (lr=0.001)

------------------------------------------------------------------------

## 🚀 Kurulum

``` bash
# Gerekli kütüphaneleri yükle
# API’yi başlat
python app.py
```

------------------------------------------------------------------------

## 📌 Notlar

-   API'ye gönderilen verilerin sırası, eğitim sırasında kullanılan
    sıralama ile uyumlu olmalıdır.
