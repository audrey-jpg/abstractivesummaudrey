import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Konfigurasi halaman harus ada di awal
st.set_page_config(page_title="Audrey-Summ: Peringkasan AI - Fine-Tuned Abstractive Training", page_icon="📄", layout="wide")

# Load model dari Hugging Face
@st.cache_resource
def load_model():
    model_name = "audreyyy/FinetunedabstractiveFLANT5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Fungsi untuk peringkasan abstraktif
def summarize_text(text, max_length, num_beams):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs.input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# UI Streamlit
st.title("🤖 Audrey-Summ: Peringkasan AI - Fine-Tuned Abstractive Training")
st.write("Masukkan teks berita dan dapatkan ringkasan otomatis dengan AI yang cepat dan akurat!")

# Deskripsi Model
st.markdown("## ℹ️ Tentang Model")
st.info(
    "**Peringkasan Abstraktif** 📝\n"
    "Menghasilkan ringkasan dengan kalimat baru yang tetap mempertahankan inti informasi dari teks asli, sehingga lebih alami, ringkas, dan mudah dipahami.\n\n"
    "Model ini merupakan hasil fine-tuning dari **FLAN-T5 Small**, sebuah model **pretrained** dari Google yang telah diadaptasi untuk peringkasan teks berita berbahasa Indonesia. "
    "Dataset yang digunakan adalah **IndoSum**, kumpulan berita dalam bahasa Indonesia. Fine-tuning dilakukan menggunakan **AutoModel** dan **AutoTokenizer** dari Hugging Face, "
    "memungkinkan AI menyajikan ringkasan yang relevan dan berkualitas tinggi.\n\n"
    "✨ Demo ini menggunakan model fine-tuned dengan metode abstraktif untuk menghasilkan ringkasan yang lebih alami dan mudah dipahami. 🚀"
)

# Input teks
text_input = st.text_area("📜 Masukkan teks berita:", height=200)

# Pengaturan panjang ringkasan & jumlah beams
col1, col2 = st.columns(2)
with col1:
    summary_length = st.slider("📏 Panjang ringkasan (karakter):", min_value=100, max_value=300, value=265, step=5)
with col2:
    num_beams = st.slider("🔍 Akurasi peringkasan (beams):", min_value=1, max_value=10, value=4, step=1)

# Tombol Ringkas
if st.button("⚡ Ringkas Sekarang"):
    if text_input.strip():
        with st.spinner("⏳ AI sedang bekerja..."):
            summary = summarize_text(text_input, summary_length, num_beams)
        st.subheader("📢 Hasil Ringkasan")
        st.success(summary)
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu!")

# Footer
st.markdown("---")
st.markdown("✨ *Oleh Audrey Roselian dibangun dengan Streamlit & Transformers dari Hugging Face* 🚀")
