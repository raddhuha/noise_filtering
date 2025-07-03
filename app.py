import streamlit as st
import torch
import torch.nn as nn
import whisper
import librosa
import soundfile as sf
import numpy as np
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64
from audio_recorder_streamlit import audio_recorder
import torchaudio

# Import untuk pretrained Demucs
from demucs import pretrained
from demucs.apply import apply_model

# Konfigurasi halaman
st.set_page_config(
    page_title="Audio Noise Filtering & ASR",
    page_icon="🎵",
    layout="wide"
)

# Cache models untuk performa
@st.cache_resource
def load_whisper_model(model_size):
    """Load Whisper model dengan caching"""
    return whisper.load_model(model_size)

@st.cache_resource
def load_demucs_model():
    """
    Load HT-Demucs model dengan caching untuk efisiensi
    """
    try:
        model = pretrained.get_model('htdemucs')
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def separate_audio(model, audio_path, output_dir):
    """
    Memisahkan audio menggunakan HT-Demucs
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to the model's expected sample rate (44100 Hz)
        if sample_rate != model.samplerate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=model.samplerate
            )
            waveform = resampler(waveform)
        
        # Ensure stereo (2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        
        # Apply model untuk memisahkan audio
        with torch.no_grad():
            sources = apply_model(model, waveform.unsqueeze(0))
        
        # sources shape: [batch, sources, channels, time]
        sources = sources.squeeze(0)  # Remove batch dimension
        
        # Simpan setiap komponen
        source_names = ['drums', 'bass', 'other', 'vocals']
        output_files = {}
        
        for i, name in enumerate(source_names):
            output_path = os.path.join(output_dir, f"{name}.mp3")
            # Convert tensor to numpy and save using torchaudio
            audio_tensor = sources[i].cpu()
            
            # Ensure the tensor is in the right format
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            torchaudio.save(
                output_path,
                audio_tensor,
                sample_rate=model.samplerate,
                format="mp3"
            )
            output_files[name] = output_path
        
        return True, output_files, None
        
    except Exception as e:
        return False, {}, str(e)

def plot_waveform_comparison(original_audio, processed_audio, sr, title_prefix="Audio"):
    """Plot perbandingan waveform sebelum dan sesudah processing"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original audio
    time_orig = np.linspace(0, len(original_audio) / sr, len(original_audio))
    ax1.plot(time_orig, original_audio, color='red', alpha=0.7)
    ax1.set_title(f"{title_prefix} - Sebelum Dibersihkan (Noisy)", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Processed audio
    time_proc = np.linspace(0, len(processed_audio) / sr, len(processed_audio))
    ax2.plot(time_proc, processed_audio, color='green', alpha=0.7)
    ax2.set_title(f"{title_prefix} - Setelah Dibersihkan (Denoised)", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    return fig

def plot_spectrogram_comparison(original_audio, processed_audio, sr, title_prefix="Audio"):
    """Plot perbandingan spectrogram sebelum dan sesudah processing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original spectrogram
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=ax1, cmap='viridis')
    ax1.set_title(f"{title_prefix} - Sebelum Dibersihkan (Noisy)", fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frekuensi (Hz)')
    ax1.set_xlabel('Waktu (s)')
    
    # Processed spectrogram
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)), ref=np.max)
    librosa.display.specshow(D_proc, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
    ax2.set_title(f"{title_prefix} - After Cleaning (Denoised)", fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frekuensi (Hz)')
    ax2.set_xlabel('Waktu (s)')
    
    plt.tight_layout()
    return fig

def process_with_pretrained_demucs(audio_path, model):
    """Proses audio dengan pretrained HT-Demucs untuk noise filtering"""
    try:
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Separate audio using the pretrained model
            success, output_files, error = separate_audio(model, audio_path, temp_dir)
            
            if not success:
                st.error(f"Error separating audio: {error}")
                return None, None
            
            # Load vocals (clean audio)
            vocals_path = output_files['vocals']
            vocals_audio, sr = librosa.load(vocals_path, sr=44100, mono=True)
            
            return vocals_audio, sr
        
    except Exception as e:
        st.error(f"Error processing with pretrained HT-Demucs: {e}")
        return None, None

def calculate_audio_metrics(original, processed, sr):
    """Hitung metrics untuk perbandingan audio"""
    try:
        # Ensure same length
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]
        
        # RMS Energy
        rms_original = np.sqrt(np.mean(original**2))
        rms_processed = np.sqrt(np.mean(processed**2))
        
        # SNR estimation (simple)
        noise_estimate = original - processed
        signal_power = np.mean(processed**2)
        noise_power = np.mean(noise_estimate**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Peak amplitude
        peak_original = np.max(np.abs(original))
        peak_processed = np.max(np.abs(processed))
        
        return {
            'rms_original': rms_original,
            'rms_processed': rms_processed,
            'snr_improvement': snr,
            'peak_original': peak_original,
            'peak_processed': peak_processed,
            'noise_reduction': 1 - (noise_power / (signal_power + 1e-10))
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None

def transcribe_audio(audio_path, model, language="auto"):
    """Transkripsi audio dengan Whisper"""
    try:
        result = model.transcribe(
            audio_path, 
            language=None if language == "auto" else language,
            task="transcribe"
        )
        return result
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def save_audio(audio, sr, format="wav"):
    """Save audio ke temporary file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
    sf.write(temp_file.name, audio, sr)
    return temp_file.name

def get_download_link(file_path, filename, text="Download"):
    """Generate download link untuk file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main App
def main():
    st.title("🎵 Audio Noise Filtering & Speech Recognition")
    st.markdown("Aplikasi untuk membersihkan noise dari audio menggunakan dan mengkonversi ke teks")
    
    # Sidebar untuk konfigurasi
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Model selection
        whisper_model_size = st.selectbox(
            "Ukuran Model Whisper",
            ["tiny", "base", "small", "medium", "large"],
            index=1
        )
        
        # Language selection
        language = st.selectbox(
            "Bahasa Audio",
            ["auto", "id", "en", "zh", "ja", "ko"],
            help="auto = deteksi otomatis"
        )
        
        # Processing options
        st.subheader("Opsi Processing")
        enable_noise_filtering = st.checkbox("Hilangkan Noise", value=True)
        normalize_audio = st.checkbox("Normalisasi Suara", value=False)
        trim_silence = st.checkbox("Hapus Suara Hening", value=False)
        
        # Visualization options
        st.subheader("Opsi Visualisasi")
        show_waveform = st.checkbox("Tampilkan Waveform", value=True)
        show_spectrogram = st.checkbox("Tampilkan Spectrogram", value=True)
        show_metrics = st.checkbox("Tampilkan Audio Metrics", value=True)
    
    # Load models
    with st.spinner("Loading models..."):
        whisper_model = load_whisper_model(whisper_model_size)
        demucs_model, demucs_error = load_demucs_model() if enable_noise_filtering else (None, None)
        
        if enable_noise_filtering:
            if demucs_model is None:
                st.error(f"❌ Error loading pretrained HT-Demucs model: {demucs_error}")
                st.stop()
    
    # Input section
    st.header("Input Audio")
    input_method = st.radio(
        "Pilih metode input:",
        ["Upload File", "Record Audio"]
    )
    
    audio_file = None
    audio_path = None
    
    if input_method == "Upload File":
        audio_file = st.file_uploader(
            "Upload file audio",
            type=["wav", "mp3", "m4a", "flac", "ogg"]
        )
        
        if audio_file:
            # Save uploaded file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_file.read())
            audio_path = temp_file.name
            
    elif input_method == "Record Audio":
        st.info("Klik tombol record untuk mulai merekam")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=5.0
        )
        
        if audio_bytes:
            # Save recorded audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_bytes)
            audio_path = temp_file.name
    
    # Processing section
    if audio_path:
        st.header("🎛️ Informasi Audio")
        
        # Display original audio info
        try:
            original_audio, sr = librosa.load(audio_path, sr=None)
            duration = len(original_audio) / sr
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Durasi", f"{duration:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col3:
                st.metric("Channels", "Mono" if len(original_audio.shape) == 1 else "Stereo")
            with col4:
                device_type = "GPU" if torch.cuda.is_available() else "CPU"
                st.metric("Device", device_type)
            
            # Play original audio
            st.subheader("🎵 Original Audio (Noisy)")
            st.audio(audio_path)
            
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            return
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Memproses audio dengan HT-Demucs..."):
                processed_audio = original_audio
                processed_sr = sr
                
                # Noise filtering dengan pretrained Demucs
                if enable_noise_filtering and demucs_model:
                    processed_audio, processed_sr = process_with_pretrained_demucs(audio_path, demucs_model)
                    if processed_audio is None:
                        st.error("❌ Gagal memproses dengan pretrained HT-Demucs")
                        return
                
                # Audio enhancement
                if normalize_audio:
                    st.info("📊 Normalizing volume...")
                    processed_audio = librosa.util.normalize(processed_audio)
                
                if trim_silence:
                    st.info("✂️ Trimming silence...")
                    processed_audio, _ = librosa.effects.trim(processed_audio, top_db=20)
                
                # Save processed audio
                processed_path = save_audio(processed_audio, processed_sr)
                
                # Results section
                st.header("📊 Hasil")
                
                # Audio comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔴 Audio Asli (Noisy)")
                    st.audio(audio_path)
                
                with col2:
                    st.subheader("🟢 Audio Bersih (Clean)")
                    st.audio(processed_path)
                    
                    # Download link
                    st.markdown(
                        get_download_link(processed_path, "denoised_audio.wav", "📥 Download Audio Bersih"),
                        unsafe_allow_html=True
                    )
                
                # Waveform comparison
                if show_waveform:
                    st.subheader("📈 Perbandingan Waveform")
                    fig_wave = plot_waveform_comparison(original_audio, processed_audio, sr, "Audio")
                    st.pyplot(fig_wave)
                
                # Spectrogram comparison
                if show_spectrogram:
                    st.subheader("🎨 Perbandingan Spectrogram")
                    try:
                        fig_spec = plot_spectrogram_comparison(original_audio, processed_audio, sr, "Audio")
                        st.pyplot(fig_spec)
                    except Exception as e:
                        st.error(f"Error Pembuatan spectrogram: {e}")
                
                # Audio metrics
                if show_metrics:
                    st.subheader("📊 Kualitas Audio Metrics")
                    metrics = calculate_audio_metrics(original_audio, processed_audio, sr)
                    
                    if metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "RMSE", 
                                f"{metrics['rms_processed']:.4f}",
                                f"{metrics['rms_processed'] - metrics['rms_original']:.4f}"
                            )
                        with col2:
                            st.metric(
                                "Peningakatan SNR", 
                                f"{metrics['snr_improvement']:.2f} dB"
                            )
                        with col3:
                            st.metric(
                                "Peak Reduction", 
                                f"{metrics['peak_processed']:.4f}",
                                f"{metrics['peak_processed'] - metrics['peak_original']:.4f}"
                            )
                        with col4:
                            st.metric(
                                "Noise Reduction", 
                                f"{metrics['noise_reduction']*100:.1f}%"
                            )
                
                # Transcription section
                st.subheader("📝 Speech Recognition")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔴 Transkripsi dari Noisy Audio:**")
                    with st.spinner("Transkripsi noisy audio..."):
                        original_transcription = transcribe_audio(audio_path, whisper_model, language)
                        if original_transcription:
                            st.text_area("Transkripsi Asli", original_transcription["text"], height=150, key="orig_transcript")
                
                with col2:
                    st.write("**🟢 Transkripsi dari Clean Audio:**")
                    with st.spinner("Transkripsi clean audio..."):
                        clean_transcription = transcribe_audio(processed_path, whisper_model, language)
                        if clean_transcription:
                            st.text_area("Transkripsi Bersih", clean_transcription["text"], height=150, key="clean_transcript")
                
                # Detailed segments
                if clean_transcription and st.expander("📋 Detailed Segments (Clean Audio)"):
                    for i, segment in enumerate(clean_transcription["segments"]):
                        confidence = segment.get('confidence', 0)
                        st.write(f"**[{segment['start']:.2f}s - {segment['end']:.2f}s]** (Confidence: {confidence:.2f}): {segment['text']}")
                
                # Download transcripts
                if clean_transcription:
                    transcript_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt")
                    transcript_file.write(f"=== ORIGINAL TRANSCRIPT (NOISY) ===\n")
                    if original_transcription:
                        transcript_file.write(original_transcription["text"])
                    transcript_file.write(f"\n\n=== CLEAN TRANSCRIPT (DENOISED) ===\n")
                    transcript_file.write(clean_transcription["text"])
                    transcript_file.close()
                    
                    st.markdown(
                        get_download_link(transcript_file.name, "transcripts_comparison.txt", "📥 Download Transkrip    "),
                        unsafe_allow_html=True
                    )
                
                # Processing summary
                st.subheader("📋 Processing Summary")
                col1, col2, col3 = st.columns(3)
                
                original_size = os.path.getsize(audio_path)
                processed_size = os.path.getsize(processed_path)
                
                with col1:
                    st.metric("Ukuran Asli", f"{original_size/1024:.1f} KB")
                with col2:
                    st.metric("Ukuran Setelah Diproses", f"{processed_size/1024:.1f} KB")
                with col3:
                    size_change = (processed_size - original_size) / original_size * 100
                    st.metric("Perubahan Ukuran", f"{size_change:+.1f}%")
                
                # Cleanup
                try:
                    os.unlink(processed_path)
                    if 'transcript_file' in locals():
                        os.unlink(transcript_file.name)
                except:
                    pass

if __name__ == "__main__":
    main()