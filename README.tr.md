# Türkçe soru-cevap embedding'leri üzerinde optimizasyon

Yerel bir Türkçe dil modeliyle örnek veri üretip bu veriden kurulan sınıflandırıcıda GD, SGD ve Adam algoritmalarını karşılaştırdığım ders çalışması.

[Notebook](optimization.ipynb) · [English and reproduction notes](README.md)

Turkish-Gemma-9B, `llama-cpp-python` üzerinden GGUF biçiminde kullanılır. Sekiz konudan soru, uyumlu cevap ve uyumsuz cevap örnekleri üretilir. `turkish-e5-large` embedding'leri ve 0.82 cosine similarity eşiğiyle benzer sorular elenir. Optimizasyon algoritmaları NumPy ile uygulanır; beş farklı başlangıç ve 100 epoch üzerinden karşılaştırılır.

Üretim hedefi 100 soru/olumlu cevap/olumsuz cevap üçlüsüdür. Her üçlü iki etiketli soru-cevap örneği verebilir; bu, 200 farklı soru anlamına gelmez.

## Yeniden çalıştırma

Depoda üretilmiş veri dosyaları ve sabitlenmiş bir ortam bulunmuyor. Üretim bölümü `dataset_v2.json` yazarken sonraki bölümler `dataset_v3.json` ve `dataset_v3_with_embeddings.json` bekliyor. Boş bir ortamda bütün hücreleri değiştirmeden çalıştırmak kayıtlı deneyi yeniden üretmez.

Yeni deney için dosya adları tutarlı hale getirilmeli, veri ve embedding'ler yeniden üretilmeli; yeni sonuçlar mevcut hücre çıktılarından ayrı kaydedilmelidir. Dokümantasyon güncellemesinde model yeniden çalıştırılmadı.

Bu çalışma küçük ve sentetik bir veri kümesindeki karşılaştırmadır; dil modeli eğitimi veya bütün problemler için genel optimizasyon sıralaması değildir.
