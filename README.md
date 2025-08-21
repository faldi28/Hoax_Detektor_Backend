# Deteksi Hoax API

## Deskripsi
Aplikasi ini adalah API untuk mendeteksi apakah suatu teks adalah hoax atau valid menggunakan model BERT untuk klasifikasi teks. Aplikasi ini dibangun menggunakan Flask dan dilengkapi dengan CORS untuk mendukung permintaan dari domain lain.

## Struktur Folder
Pastikan Anda memiliki struktur folder sebagai berikut:

/project_folder 
├── app.py 
└── /model 
    ├── model.safetensors 
    ├── pytorch_model.bin 
    └── config.json


## Persyaratan
Sebelum menjalankan aplikasi, pastikan Anda telah menginstal Python dan pustaka berikut:
- Flask
- torch
- transformers
- langdetect

Anda dapat menginstal pustaka yang diperlukan dengan menggunakan pip:
```bash
pip install flask torch transformers langdetect

## Cara Menjalankan Aplikasi
Buka terminal dan navigasikan ke folder proyek.
Jalankan perintah berikut:python app.py
Aplikasi akan berjalan di http://0.0.0.0:5000. Anda dapat mengakses endpoint / untuk memeriksa status server atau /predict untuk melakukan prediksi.

## Endpoint API
GET /: Memeriksa status server.
Response: Menampilkan pesan bahwa server berjalan dan memberikan informasi tentang endpoint /predict.
POST /predict: Mengirimkan teks untuk dianalisis.
Format Permintaan:
{
  "text": "Teks yang akan dianalisis",
  "title": "Judul berita"
}

## Response:
Jika berhasil, akan mengembalikan prediksi dan judul.
Jika terjadi kesalahan, akan mengembalikan kode kesalahan dan pesan yang sesuai.

## Fungsi Utama
Inisialisasi Aplikasi: Mengatur aplikasi Flask dan mengonfigurasi CORS.
Pemuatan Model: Memuat model dan tokenizer dari file yang disimpan di direktori 'model'.
Pra-pemrosesan Teks: Membersihkan teks dengan mengubahnya menjadi huruf kecil, menghapus URL, karakter berulang, dan tanda baca.
Prediksi: Mengambil teks yang telah diproses, melakukan tokenisasi, dan menggunakan model untuk memprediksi apakah teks tersebut hoax atau valid.

## Catatan
Pastikan semua file model telah disalin ke folder 'model' agar aplikasi dapat berjalan dengan baik.
Jika model tidak ditemukan, server akan berjalan dengan fungsionalitas terbatas
undefined