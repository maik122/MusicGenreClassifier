# app.py
# Run with: streamlit run app.py

import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import librosa
import librosa.display
from src.predict import load_models, predict_genre
from src.config import GENRES
import download_models
download_models.download_models_if_missing()


# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Genre.AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# Global styles — dark, music-native aesthetic
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ---- Reset & base ---- */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6f0;
}

.stApp {
    background: #0a0a0f;
}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ---- Main container ---- */
.block-container {
    max-width: 1100px;
    padding: 2rem 2rem 4rem;
}

/* ---- Hero ---- */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f5f3ff 0%, #a78bfa 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}

.hero-sub {
    color: #71717a;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

/* ---- Genre pills ---- */
.genre-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-bottom: 2.5rem;
}

.genre-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #52525b;
    border: 1px solid #27272a;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    letter-spacing: 0.05em;
}

/* ---- Upload zone ---- */
[data-testid="stFileUploader"] {
    background: #111118 !important;
    border: 1.5px dashed #27272a !important;
    border-radius: 16px !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7c3aed !important;
}

[data-testid="stFileUploader"] label {
    color: #52525b !important;
    font-size: 0.9rem !important;
}

/* ---- Predict button ---- */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 100px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 24px rgba(124, 58, 237, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 36px rgba(124, 58, 237, 0.5) !important;
}

/* ---- Result card ---- */
.result-card {
    background: linear-gradient(135deg, #111118, #18181f);
    border: 1px solid #27272a;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7c3aed, #a78bfa, #7c3aed);
}

.result-genre {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f5f3ff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.result-desc {
    color: #71717a;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.confidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #52525b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.confidence-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #18181f;
}

/* ---- Stat cards ---- */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    flex: 1;
    background: #111118;
    border: 1px solid #1c1c24;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #52525b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e6f0;
}

/* ---- Top 3 predictions ---- */
.top-pred {
    background: #111118;
    border: 1px solid #1c1c24;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.top-pred-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3f3f46;
    width: 1.5rem;
}

.top-pred-genre {
    font-weight: 700;
    font-size: 1rem;
    flex: 1;
    padding-left: 0.75rem;
}

.top-pred-bar-wrap {
    flex: 2;
    height: 4px;
    background: #1c1c24;
    border-radius: 100px;
    margin: 0 1rem;
    overflow: hidden;
}

.top-pred-bar {
    height: 100%;
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    border-radius: 100px;
}

.top-pred-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #a78bfa;
    text-align: right;
    min-width: 3.5rem;
}

/* ---- Audio player ---- */
audio {
    width: 100%;
    border-radius: 12px;
    filter: invert(1) hue-rotate(180deg);
}

/* ---- Divider ---- */
hr { border-color: #18181f !important; }

/* ---- Spinner ---- */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid #18181f !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Load models
# -------------------------------------------------------
@st.cache_resource
def get_models():
    return load_models()

rf, scaler, le = get_models()

# -------------------------------------------------------
# Genre metadata
# -------------------------------------------------------
GENRE_INFO = {
    "blues":     "Guitar-driven expressive music rooted in African-American tradition",
    "classical": "Orchestral acoustic music with complex harmonic structure",
    "country":   "Folk-inspired American music with storytelling lyrics",
    "disco":     "High-energy dance music with strong four-on-the-floor rhythm",
    "hiphop":    "Rhythmic vocal-driven music over sampled or synthesised beats",
    "jazz":      "Improvisational music with complex chord progressions",
    "metal":     "High-intensity music with heavy distorted guitars",
    "pop":       "Hook-driven mainstream commercial music",
    "reggae":    "Offbeat rhythm music originating in Jamaica",
    "rock":      "Electric guitar-driven music built on blues foundations",
}

GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠", "disco": "🪩",
    "hiphop": "🎤", "jazz": "🎷", "metal": "🤘", "pop": "🎶",
    "reggae": "🌿", "rock": "⚡",
}

# -------------------------------------------------------
# Hero
# -------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">Random Forest · GTZAN · 77% Accuracy</div>
    <div class="hero-title">Genre.AI</div>
    <div class="hero-sub">Drop any song and the model reads its acoustic fingerprint to identify the genre.</div>
    <div class="genre-pills">
        <span class="genre-pill">Blues</span>
        <span class="genre-pill">Classical</span>
        <span class="genre-pill">Country</span>
        <span class="genre-pill">Disco</span>
        <span class="genre-pill">Hip-hop</span>
        <span class="genre-pill">Jazz</span>
        <span class="genre-pill">Metal</span>
        <span class="genre-pill">Pop</span>
        <span class="genre-pill">Reggae</span>
        <span class="genre-pill">Rock</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Upload + predict — two column layout
# -------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-header">Upload Track</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop a .wav or .mp3 file",
        type=["wav", "mp3"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.audio(uploaded_file)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Analyse Genre →")
    else:
        st.markdown("""
        <div style="color:#3f3f46; font-size:0.85rem; margin-top:1rem; font-family:'DM Mono',monospace;">
        Supports .wav and .mp3 · up to 200MB
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

with right:
    if uploaded_file and predict_btn:
        with st.spinner("Reading acoustic features..."):
            with tempfile.NamedTemporaryFile(
                suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            result = predict_genre(tmp_path, rf, scaler, le)
            y, sr = librosa.load(tmp_path)

        predicted = result["genre"]
        probs = result["probabilities"]
        top_conf = probs[predicted]

        # --- Result card ---
        emoji = GENRE_EMOJI.get(predicted, "🎵")
        desc  = GENRE_INFO.get(predicted, "")

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem;margin-bottom:0.5rem">{emoji}</div>
            <div class="result-genre">{predicted.capitalize()}</div>
            <div class="result-desc">{desc}</div>
            <div class="confidence-label">Confidence</div>
            <div class="confidence-value">{top_conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div style="
            height: 280px;
            background: #0d0d14;
            border: 1px dashed #1c1c24;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #27272a;
            font-family: 'DM Mono', monospace;
            font-size: 0.8rem;
            letter-spacing: 0.1em;
        ">PREDICTION WILL APPEAR HERE</div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# Results section (only after prediction)
# -------------------------------------------------------
if uploaded_file and predict_btn:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    # --- Audio stats ---
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    st.markdown('<div class="section-header">Track Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Duration</div>
            <div class="stat-value">{duration:.1f}s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Sample Rate</div>
            <div class="stat-value">{sr//1000}kHz</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Tempo</div>
            <div class="stat-value">{float(tempo):.0f} BPM</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Samples</div>
            <div class="stat-value">{len(y)//1000}K</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Three columns: top preds / bar chart / radar ---
    c1, c2, c3 = st.columns([1, 1.3, 1], gap="large")

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    with c1:
        st.markdown('<div class="section-header">Top Predictions</div>', unsafe_allow_html=True)
        for i, (genre, prob) in enumerate(sorted_probs[:5]):
            bar_w = int(prob * 100)
            rank_color = ["#a78bfa", "#7c3aed", "#4c1d95", "#2d1b69", "#1e1040"][i]
            st.markdown(f"""
            <div class="top-pred">
                <div class="top-pred-rank" style="color:{rank_color}">0{i+1}</div>
                <div class="top-pred-genre">{genre.capitalize()}</div>
                <div class="top-pred-bar-wrap">
                    <div class="top-pred-bar" style="width:{bar_w}%"></div>
                </div>
                <div class="top-pred-pct">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
        genres_sorted = [g for g, _ in sorted_probs]
        values_sorted = [v for _, v in sorted_probs]
        colours = ["#7c3aed" if g == predicted else "#1c1c2e" for g in genres_sorted]
        edge_colours = ["#a78bfa" if g == predicted else "#27272a" for g in genres_sorted]

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#111118')
        ax.set_facecolor('#111118')

        bars = ax.barh(
            genres_sorted, values_sorted,
            color=colours, edgecolor=edge_colours, linewidth=0.8, height=0.65
        )

        ax.set_xlim(0, max(values_sorted) * 1.25)
        ax.tick_params(colors='#52525b', labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.tick_params(left=False)

        for bar, val in zip(bars, values_sorted):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va='center', color='#71717a', fontsize=8,
                fontfamily='monospace'
            )

        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-header">Genre Radar</div>', unsafe_allow_html=True)
        labels = list(probs.keys())
        stats  = list(probs.values())
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        stats_plot  = stats  + [stats[0]]
        angles_plot = angles + [angles[0]]

        fig2 = plt.figure(figsize=(4, 4))
        fig2.patch.set_facecolor('#111118')
        ax2 = fig2.add_subplot(111, polar=True)
        ax2.set_facecolor('#111118')

        ax2.plot(angles_plot, stats_plot, color='#7c3aed', linewidth=2)
        ax2.fill(angles_plot, stats_plot, color='#7c3aed', alpha=0.2)

        ax2.set_xticks(angles)
        ax2.set_xticklabels(
            [l.capitalize() for l in labels],
            color='#52525b', size=7.5
        )
        ax2.set_yticklabels([])
        ax2.grid(color='#1c1c24', linewidth=0.5)
        ax2.spines['polar'].set_color('#1c1c24')

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # --- Waveform + Spectrogram ---
    st.markdown('<hr>', unsafe_allow_html=True)
    w1, w2 = st.columns(2, gap="large")

    with w1:
        st.markdown('<div class="section-header">Waveform</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 2.5))
        fig3.patch.set_facecolor('#111118')
        ax3.set_facecolor('#111118')
        librosa.display.waveshow(y, sr=sr, ax=ax3, color='#7c3aed', alpha=0.9)
        ax3.set_xlabel('')
        ax3.tick_params(colors='#3f3f46', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_color('#1c1c24')
        plt.tight_layout(pad=0.3)
        st.pyplot(fig3, use_container_width=True)

    with w2:
        st.markdown('<div class="section-header">Spectrogram</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 2.5))
        fig4.patch.set_facecolor('#111118')
        ax4.set_facecolor('#111118')
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis='time', y_axis='log',
            ax=ax4, cmap='Purples'
        )
        ax4.tick_params(colors='#3f3f46', labelsize=8)
        for spine in ax4.spines.values():
            spine.set_color('#1c1c24')
        cb = fig4.colorbar(img, ax=ax4, format='%+2.0f dB')
        cb.ax.tick_params(colors='#3f3f46', labelsize=7)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig4, use_container_width=True)

    # --- Footer ---
    st.markdown("""
    <div style="text-align:center; margin-top:3rem; color:#27272a; font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.1em;">
        GENRE.AI · RANDOM FOREST · GTZAN DATASET · 77% ACCURACY
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0">
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#52525b;text-transform:uppercase;margin-bottom:1.5rem">Project Info</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model**")
    st.caption("Random Forest — 200 Trees")

    st.markdown("**Dataset**")
    st.caption("GTZAN — 1,000 songs × 10 genres")

    st.markdown("**Accuracy**")
    st.caption("77% on held-out test set")

    st.markdown("**Features**")
    st.caption("MFCCs, chroma, spectral centroid, rolloff, ZCR, tempo, harmony")

    st.divider()

    st.markdown("**Genres**")
    for g in sorted(GENRES):
        e = GENRE_EMOJI.get(g, "•")
        st.caption(f"{e} {g.capitalize()}")