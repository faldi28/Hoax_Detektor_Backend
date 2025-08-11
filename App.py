import flask
from flask import request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import re
import sys
from langdetect import detect, LangDetectException
import os # os masih digunakan untuk mengambil environment variable jika ada

# --- Inisialisasi Aplikasi Flask dan CORS ---
app = flask.Flask(__name__)
CORS(app)
# SARAN: Untuk production di Vercel, nilai DEBUG tidak terlalu berpengaruh,
# tapi untuk production di server sendiri, sebaiknya diatur ke False.
app.config["DEBUG"] = True


# --- Konfigurasi dan Pemuatan Model dari Hugging Face Hub ---

# <-- GANTI INI dengan ID model Anda di Hugging Face Hub (misal: "username/nama-model")
MODEL_NAME = "faldi05/model_hoax"

model = None
tokenizer = None

def load_model_and_tokenizer():
    """
    Fungsi ini memuat tokenizer dan model langsung dari Hugging Face Hub
    saat aplikasi pertama kali dijalankan.
    """
    global model, tokenizer
    try:
        print(f"Backend: Mencoba memuat model dari Hugging Face Hub: {MODEL_NAME}", file=sys.stdout)
        
        # Jika model Anda private, pastikan HUGGING_FACE_HUB_TOKEN ada di Environment Variables Vercel
        # token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, token=token)
        # model = BertForSequenceClassification.from_pretrained(MODEL_NAME, token=token)
        
        # Jika model public, token tidak diperlukan
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Pindahkan model ke GPU jika tersedia (di Vercel biasanya hanya CPU)
        if torch.cuda.is_available():
            model.to("cuda")
        
        model.eval() # Set model ke mode evaluasi/inferensi
        print(">>> Backend: Model dan tokenizer berhasil dimuat dari Hugging Face Hub.", file=sys.stdout)
        return True
        
    except Exception as e:
        print(f"Backend: Gagal memuat model dari Hugging Face Hub. Error: {e}", file=sys.stderr)
        print(">>> PASTIKAN NAMA MODEL SUDAH BENAR DAN REPOSITORY MODEL DI HUGGING FACE ADALAH 'PUBLIC'.", file=sys.stderr)
        return False

# Muat model saat server dinyalakan
is_model_loaded = load_model_and_tokenizer()


# --- Fungsi Pra-pemrosesan Teks ---
def clean_text_minimal(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # Ubah ke huruf kecil
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Hapus URL
    text = re.sub(r'(.)\1{2,}', r'\1', text) # Hapus karakter berulang (e.g., "haaiii" -> "haai")
    text = re.sub(r'[^\w\s]', '', text) # Hapus semua tanda baca
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

def truncate_head_tail(text, tokenizer, max_length=512):
    """
    Memotong teks panjang dengan mempertahankan bagian awal (head) dan akhir (tail),
    sesuai dengan strategi pada saat training model.
    """
    tokens = tokenizer.tokenize(text)
    # Kurangi 2 untuk memperhitungkan token spesial [CLS] dan [SEP]
    if len(tokens) > max_length - 2:
        head_len = 128
        tail_len = max_length - head_len - 2
        
        head_tokens = tokens[:head_len]
        tail_tokens = tokens[-tail_len:]
        
        return tokenizer.convert_tokens_to_string(head_tokens + tail_tokens)
    return text


# --- Fungsi Prediksi Inti ---
def predict(text):
    if not is_model_loaded:
        # Jika model gagal dimuat, kembalikan error
        return {"error_code": "MODEL_ERROR", "message": "Model tidak dapat dimuat di server."}, 503

    # 1. Bersihkan teks input
    cleaned_text = clean_text_minimal(text)

    # 2. Potong teks jika terlalu panjang
    truncated_text = truncate_head_tail(cleaned_text, tokenizer)

    # 3. Tokenisasi teks untuk input model
    inputs = tokenizer(
        truncated_text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    )

    # Pindahkan input tensors ke device yang sama dengan model (CPU/GPU)
    device = model.device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Lakukan prediksi tanpa menghitung gradien
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()

    # Ubah hasil probabilitas ke list Python
    probabilities = probabilities.cpu().tolist()
    
    # Tentukan label berdasarkan probabilitas tertinggi
    # Asumsi: 0 = Hoax, 1 = Valid
    predicted_class_id = probabilities.index(max(probabilities))
    label = "Valid" if predicted_class_id == 1 else "Hoax"
    
    return {"prediction": label}, 200


# --- Endpoint API ---
@app.route('/', methods=['GET'])
def home():
    """Endpoint dasar untuk mengecek apakah server berjalan."""
    return "<h1>API Deteksi Hoax</h1><p>Server berjalan. Gunakan endpoint /predict dengan metode POST.</p>"

@app.route('/predict', methods=['POST'])
def predict_api():
    """Endpoint utama untuk menerima teks dan memberikan prediksi hoax/valid."""
    if not is_model_loaded:
        return jsonify({"error_code": "MODEL_UNAVAILABLE", "message": "Model AI tidak tersedia di server saat ini."}), 503

    # Validasi input dari request
    if not request.json or 'text' not in request.json or 'title' not in request.json:
        return jsonify({"error_code": "INVALID_REQUEST", "message": "Permintaan harus dalam format JSON dan berisi 'text' dan 'title'."}), 400

    text_to_predict = request.json['text']
    title_from_request = request.json['title']

    if not isinstance(text_to_predict, str) or not text_to_predict.strip():
        return jsonify({"error_code": "INVALID_TEXT", "message": "Input 'text' tidak boleh kosong."}), 400

    # Validasi panjang teks (minimal 50 kata)
    if len(text_to_predict.split()) < 50:
        return jsonify({"error_code": "TEXT_TOO_SHORT", "message": "Konten teks terlalu pendek untuk dianalisis sebagai berita (min. 50 kata)."}), 400

    # Validasi bahasa (harus Bahasa Indonesia)
    try:
        language = detect(text_to_predict)
        if language != 'id':
            return jsonify({"error_code": "NOT_INDONESIAN", "message": "Teks yang dianalisis harus dalam Bahasa Indonesia."}), 400
    except LangDetectException:
        return jsonify({"error_code": "LANGUAGE_UNKNOWN", "message": "Bahasa dari teks tidak dapat dideteksi."}), 400

    # Lakukan prediksi jika semua validasi lolos
    try:
        result, status_code = predict(text_to_predict)

        if status_code != 200:
            return jsonify(result), status_code

        # Format respons akhir sesuai keinginan
        final_response = {
            "prediction": result.get("prediction"),
            "title": title_from_request 
        }

        return jsonify(final_response), 200
        
    except Exception as e:
        # Tangani error tak terduga selama proses prediksi
        print(f"Backend: Error saat prediksi: {e}", file=sys.stderr)
        return jsonify({"error_code": "INTERNAL_SERVER_ERROR", "message": "Terjadi kesalahan internal saat melakukan prediksi."}), 500

# --- Menjalankan Server ---
# Bagian ini akan berjalan jika Anda menjalankan skrip secara lokal (python App.py)
# Vercel akan mengimpor 'app' dan menjalankannya sendiri.
if __name__ == '__main__':
    # host='0.0.0.0' membuat server bisa diakses dari luar container/jaringan lokal
    app.run(host='0.0.0.0', port=5000, debug=True)