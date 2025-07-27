import flask
from flask import request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import re
import os
import sys
from langdetect import detect, LangDetectException
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# --- Inisialisasi Aplikasi Flask dan CORS ---
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

# --- (BARU) Mengunduh komponen NLTK yang diperlukan ---
# Ini hanya perlu dijalankan sekali saat server pertama kali dimulai.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print(">>> Backend: Mengunduh tokenizer 'punkt' dari NLTK...", file=sys.stdout)
    nltk.download('punkt')
    print(">>> Backend: Unduhan 'punkt' selesai.", file=sys.stdout)


# --- Konfigurasi dan Pemuatan Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    print(f"Backend: Mencoba memuat model dari: {MODEL_PATH}", file=sys.stdout)
    model_file_safetensors = os.path.join(MODEL_PATH, 'model.safetensors')
    model_file_bin = os.path.join(MODEL_PATH, 'pytorch_model.bin')
    config_file = os.path.join(MODEL_PATH, 'config.json')

    if not os.path.exists(config_file) or not (os.path.exists(model_file_safetensors) or os.path.exists(model_file_bin)):
        print(">>> PERINGATAN: File model ('model.safetensors' atau 'pytorch_model.bin') tidak ditemukan.", file=sys.stdout)
        print(f">>> Pastikan Anda telah menyalin semua file model ke folder '{MODEL_PATH}'.", file=sys.stdout)
        print(">>> Server akan berjalan dengan fungsionalitas terbatas.", file=sys.stdout)
        return False
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        print(">>> Backend: Model dan tokenizer berhasil dimuat.", file=sys.stdout)
        return True
    except Exception as e:
        print(f"Backend: Gagal memuat model. Error: {e}", file=sys.stderr)
        return False

is_model_loaded = load_model_and_tokenizer()


# --- Fungsi Pra-pemrosesan (DISESUAIKAN DENGAN COLAB) ---
def preprocess_for_bert(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|httpsS+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'([.?,!])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def truncate_head_tail(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        head_len = 128
        tail_len = max_length - head_len - 2
        head_tokens = tokens[:head_len]
        tail_tokens = tokens[-tail_len:]
        return tokenizer.convert_tokens_to_string(head_tokens + tail_tokens)
    return text

# --- (BARU) Fungsi Validasi Struktur Artikel ---
def is_likely_article(text, min_sentences=3, min_avg_words=8):
    """
    Menganalisis apakah teks memiliki struktur seperti artikel.
    Sebuah artikel seharusnya memiliki beberapa kalimat dengan panjang yang wajar.
    """
    try:
        # Memecah teks menjadi kalimat
        sentences = sent_tokenize(text)
        
        # Cek jumlah kalimat minimum
        if len(sentences) < min_sentences:
            print(f"Backend: Teks ditolak karena jumlah kalimat ({len(sentences)}) < minimum ({min_sentences}).", file=sys.stdout)
            return False

        # Menghitung total kata dan rata-rata panjang kalimat
        total_words = len(word_tokenize(text))
        avg_words_per_sentence = total_words / len(sentences)

        # Cek rata-rata kata per kalimat
        if avg_words_per_sentence < min_avg_words:
            print(f"Backend: Teks ditolak karena rata-rata kata per kalimat ({avg_words_per_sentence:.2f}) < minimum ({min_avg_words}).", file=sys.stdout)
            return False
            
        print("Backend: Teks lolos validasi struktur artikel.", file=sys.stdout)
        return True
    except Exception as e:
        print(f"Backend: Error saat validasi artikel: {e}", file=sys.stderr)
        return False # Jika ada error, anggap bukan artikel

# --- Fungsi Prediksi ---
def predict(text):
    if not is_model_loaded:
        return {"error_code": "MODEL_ERROR", "message": "Model tidak dapat dimuat di server."}, 503

    processed_text = preprocess_for_bert(text)
    final_text = truncate_head_tail(processed_text, tokenizer)

    inputs = tokenizer(
        final_text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    )
    
    device = model.device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()

    probabilities = probabilities.cpu().tolist()
    
    predicted_class_id = probabilities.index(max(probabilities))
    label = "Valid" if predicted_class_id == 1 else "Hoax"
    
    return {"prediction": label}, 200

# --- Endpoint API (DIPERBARUI) ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>API Deteksi Hoax</h1><p>Server berjalan. Gunakan endpoint /predict.</p>"

@app.route('/predict', methods=['POST'])
def predict_api():
    if not is_model_loaded:
        return jsonify({"error_code": "MODEL_UNAVAILABLE", "message": "Model AI tidak tersedia di server saat ini."}), 503

    if not request.json or 'text' not in request.json:
        return jsonify({"error_code": "INVALID_REQUEST", "message": "Permintaan harus dalam format JSON dan berisi 'text'"}), 400

    text_to_predict = request.json['text']

    if not text_to_predict or not isinstance(text_to_predict, str):
        return jsonify({"error_code": "INVALID_TEXT", "message": "Input 'text' tidak boleh kosong."}), 400

    if len(text_to_predict.split()) < 50:
        return jsonify({"error_code": "TEXT_TOO_SHORT", "message": "Konten teks terlalu pendek untuk dianalisis sebagai berita."}), 400

    try:
        language = detect(text_to_predict)
        if language != 'id':
            return jsonify({"error_code": "NOT_INDONESIAN", "message": "Teks yang dianalisis harus dalam Bahasa Indonesia."}), 400
    except LangDetectException:
        return jsonify({"error_code": "LANGUAGE_UNKNOWN", "message": "Bahasa dari teks tidak dapat dideteksi."}), 400
    
    # --- (BARU) Memanggil fungsi validasi struktur artikel ---
    if not is_likely_article(text_to_predict):
        return jsonify({"error_code": "NOT_AN_ARTICLE", "message": "Konten tidak tampak seperti artikel berita yang koheren."}), 400

    try:
        result, status_code = predict(text_to_predict)
        return jsonify(result), status_code
    except Exception as e:
        print(f"Backend: Error saat prediksi: {e}", file=sys.stderr)
        return jsonify({"error_code": "INTERNAL_SERVER_ERROR", "message": "Terjadi kesalahan internal saat melakukan prediksi."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)