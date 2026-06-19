import os
import numpy as np
from PIL import Image
import gradio as gr

# Konfigurasi model dan parameter
MODEL_SAVE_PATH = "model_waste_cnn.keras"
IMAGE_SIZE = (150, 150)

# Coba import tensorflow secara opsional
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    print("[PERINGATAN] TensorFlow tidak terdeteksi di lingkungan Python lokal.")
    print("[INFO] Aplikasi Gradio akan berjalan dalam MODE DEMO/SIMULASI.")

def simulate_prediction(img):
    """
    Simulasi klasifikasi gambar tanpa TensorFlow menggunakan analisis warna gambar (PIL).
    """
    img_small = img.resize((10, 10))
    pixels = list(img_small.getdata())
    
    avg_r = sum(p[0] for p in pixels) / 100
    avg_g = sum(p[1] for p in pixels) / 100
    avg_b = sum(p[2] for p in pixels) / 100
    
    if avg_g > avg_b + 5 and avg_g > 60:
        prob_recyclable = 0.15 + (avg_b / 255.0) * 0.1
    elif avg_r > avg_b + 15 and avg_g > avg_b + 10:
        prob_recyclable = 0.20 + (avg_b / 255.0) * 0.1
    elif avg_b > avg_r + 5 or (avg_r > 200 and avg_g > 200 and avg_b > 200):
        prob_recyclable = 0.80 + (avg_b / 255.0) * 0.15
    else:
        seed_val = int(avg_r + avg_g * 2 + avg_b * 3) % 1000
        np.random.seed(seed_val)
        prob_recyclable = np.random.uniform(0.35, 0.75)
        
    prob_recyclable = max(0.05, min(0.95, prob_recyclable))
    prob_organic = 1.0 - prob_recyclable
    return prob_recyclable, prob_organic

# Fungsi untuk melakukan prediksi gambar
def classify_waste(input_image):
    try:
        use_simulation = True
        mode_desc = ""
        
        if HAS_TENSORFLOW and os.path.exists(MODEL_SAVE_PATH):
            try:
                # Load model
                model = tf.keras.models.load_model(MODEL_SAVE_PATH)
                
                # Preprocessing gambar (resize dan konversi)
                img = input_image.resize(IMAGE_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) # batch dimension
        
                # Prediksi
                prediction = model.predict(img_array, verbose=0)[0][0]
                prob_recyclable = float(prediction)
                prob_organic = float(1 - prediction)
                use_simulation = False
            except Exception as e:
                print(f"[WARNING] Gagal memprediksi dengan model: {e}")
                
        if use_simulation:
            prob_recyclable, prob_organic = simulate_prediction(input_image)
            if not HAS_TENSORFLOW:
                mode_desc = " [Simulasi - TensorFlow Tidak Ada]"
            else:
                mode_desc = " [Simulasi - Model Belum Dilatih]"
        
        # Tentukan hasil keputusan label teks
        if prob_recyclable >= 0.5:
            decision = f"Hasil: ANORGANIK / DAUR ULANG (Recyclable){mode_desc} - Keyakinan: {prob_recyclable * 100:.2f}%"
        else:
            decision = f"Hasil: ORGANIK{mode_desc} - Keyakinan: {prob_organic * 100:.2f}%"
            
        outputs = {
            "Organic (Organik)": prob_organic,
            "Recyclable (Anorganik/Daur Ulang)": prob_recyclable
        }
        return outputs, decision
        
    except Exception as e:
        return {"Error": 1.0}, f"Terjadi kesalahan saat memproses gambar: {str(e)}"

# Membuat Antarmuka Gradio dengan Desain Modern
theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate"
)

app = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Unggah Foto Sampah Anda"),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilitas Klasifikasi"),
        gr.Textbox(label="Hasil Keputusan Akhir")
    ],
    title="🟢 Sistem Deteksi dan Pemilah Sampah Organik & Anorganik",
    description="Proyek Kecerdasan Buatan berbasis Deep Learning (CNN) untuk mengklasifikasikan sampah secara otomatis ke dalam kategori Organik dan Anorganik (Daur Ulang).",
    article="<p style='text-align: center;'>Tugas Proyek Semester 6 - Deep Learning Image Classification</p>",
    theme=theme
)

if __name__ == "__main__":
    app.launch(share=True)
