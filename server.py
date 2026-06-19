import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app) # Mengaktifkan CORS untuk seluruh rute

# Konfigurasi model dan parameter
MODEL_SAVE_PATH = "model_waste_cnn.keras"
IMAGE_SIZE = (150, 150)

# Coba import tensorflow secara opsional untuk menghindari error di python lokal tanpa tensorflow
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    print("[PERINGATAN] TensorFlow tidak terdeteksi di lingkungan Python lokal.")
    print("[INFO] Server Flask akan berjalan dalam MODE DEMO/SIMULASI.")
    print("[INFO] Anda tetap dapat membuka web, mengunggah gambar, dan menguji UI.")

# Load model secara global jika TensorFlow tersedia dan model sudah dilatih
model = None
if HAS_TENSORFLOW and os.path.exists(MODEL_SAVE_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("[INFO] Model Keras berhasil dimuat.")
    except Exception as e:
        print(f"[EROR] Gagal memuat model: {e}")

def simulate_prediction(img):
    """
    Simulasi klasifikasi gambar tanpa TensorFlow menggunakan analisis warna gambar (PIL).
    Membuat hasil simulasi terasa realistis dan responsif terhadap gambar yang diunggah.
    """
    # Resize gambar ke ukuran kecil untuk efisiensi komputasi
    img_small = img.resize((10, 10))
    pixels = list(img_small.getdata())
    
    avg_r = sum(p[0] for p in pixels) / 100
    avg_g = sum(p[1] for p in pixels) / 100
    avg_b = sum(p[2] for p in pixels) / 100
    
    # Heuristik sederhana:
    # - Warna hijau/cokelat dominan cenderung Organik (sampah organik, dedaunan, makanan).
    # - Warna biru, abu-abu logam, putih terang, kontras tinggi cenderung Daur Ulang/Anorganik.
    if avg_g > avg_b + 5 and avg_g > 60:
        # Dominan hijau -> Organik
        prob_recyclable = 0.15 + (avg_b / 255.0) * 0.1
    elif avg_r > avg_b + 15 and avg_g > avg_b + 10:
        # Dominan cokelat/kuning -> Organik
        prob_recyclable = 0.20 + (avg_b / 255.0) * 0.1
    elif avg_b > avg_r + 5 or (avg_r > 200 and avg_g > 200 and avg_b > 200):
        # Dominan biru atau putih terang -> Daur ulang
        prob_recyclable = 0.80 + (avg_b / 255.0) * 0.15
    else:
        # Berdasarkan seed yang konsisten untuk gambar yang sama
        seed_val = int(avg_r + avg_g * 2 + avg_b * 3) % 1000
        np.random.seed(seed_val)
        prob_recyclable = np.random.uniform(0.35, 0.75)
        
    prob_recyclable = max(0.05, min(0.95, prob_recyclable))
    prob_organic = 1.0 - prob_recyclable
    return prob_recyclable, prob_organic

@app.route('/')
def home():
    status_mode = "Real CNN Model" if model is not None else "Simulation"
    return f"Waste Classifier API is running! [Mode: {status_mode}]"

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada berkas gambar yang dikirim."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong."}), 400
        
    try:
        # Preprocessing gambar
        img = Image.open(file.stream).convert('RGB')
        
        # Cek apakah menggunakan model nyata atau simulasi
        use_simulation = True
        mode_desc = ""
        
        if HAS_TENSORFLOW:
            # Lazy loading model jika belum termuat
            if model is None and os.path.exists(MODEL_SAVE_PATH):
                try:
                    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
                except Exception as e:
                    print(f"[WARNING] Gagal memuat model: {e}")
            
            if model is not None:
                use_simulation = False
                img_resized = img.resize(IMAGE_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) # batch dimension
                
                # Prediksi real
                prediction = model.predict(img_array, verbose=0)[0][0]
                prob_recyclable = float(prediction)
                prob_organic = float(1 - prediction)
        
        if use_simulation:
            prob_recyclable, prob_organic = simulate_prediction(img)
            if not HAS_TENSORFLOW:
                mode_desc = " [Mode Simulasi - TensorFlow Tidak Terdeteksi]"
            elif not os.path.exists(MODEL_SAVE_PATH):
                mode_desc = " [Mode Simulasi - Model Belum Dilatih]"
            else:
                mode_desc = " [Mode Simulasi - Model Gagal Dimuat]"
        
        if prob_recyclable >= 0.5:
            class_label = "Anorganik / Daur Ulang (Recyclable)" + mode_desc
            confidence = prob_recyclable * 100
            color = "var(--accent-rec)"
            icon = "♻️"
            desc = "Sampah anorganik seperti botol plastik, kertas, kardus, gelas kaca, atau kaleng logam. Harap bersihkan dahulu sebelum dibuang ke wadah daur ulang."
        else:
            class_label = "Organik (Organic)" + mode_desc
            confidence = prob_organic * 100
            color = "var(--accent-org)"
            icon = "🍎"
            desc = "Sampah organik seperti sisa makanan, kulit buah, sayuran busuk, daun-daun kering, atau cangkang telur. Dapat diolah lebih lanjut menjadi pupuk kompos alami."
            
        return jsonify({
            "class": class_label,
            "confidence": round(confidence, 2),
            "organic_prob": round(prob_organic * 100, 2),
            "recyclable_prob": round(prob_recyclable * 100, 2),
            "description": desc,
            "color": color,
            "icon": icon
        })
        
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500

if __name__ == '__main__':
    # Pastikan folder templates ada
    os.makedirs('templates', exist_ok=True)
    port = int(os.environ.get("PORT", 7860))
    print(f"[RUNNING] Server Flask berjalan di http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
