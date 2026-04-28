import streamlit as st
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer as KerasTokenizer


st.set_page_config(
    page_title="WordFlow",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #0a0a0f !important;
    font-family: 'Outfit', sans-serif !important;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -20%; left: -10%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(120,80,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: drift1 12s ease-in-out infinite alternate;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: -10%; right: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,200,180,0.14) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: drift2 15s ease-in-out infinite alternate;
}

@keyframes drift1 {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(60px,80px) scale(1.15); }
}
@keyframes drift2 {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(-40px,-60px) scale(1.1); }
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

.block-container {
    max-width: 660px !important;
    padding: 72px 20px 80px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ── HERO ───────────────────────────────── */
.hero-wrap { text-align: center; margin-bottom: 40px; }

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    border-radius: 100px;
    padding: 5px 14px 5px 10px;
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 22px;
}

.hero-chip-dot {
    width: 6px; height: 6px;
    background: #7c5cfc;
    border-radius: 50%;
    box-shadow: 0 0 8px #7c5cfc;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(0.8); }
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(52px, 10vw, 80px);
    font-weight: 400;
    letter-spacing: -2px;
    line-height: 0.95;
    color: #ffffff;
    margin-bottom: 14px;
}

.hero-title em {
    font-style: italic;
    background: linear-gradient(135deg, #7c5cfc 0%, #00c8b4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 16px;
    font-weight: 300;
    color: rgba(255,255,255,0.38);
    letter-spacing: 0.01em;
    line-height: 1.6;
    margin-bottom: 0;
}

/* ── MODE TOGGLE ─────────────────────────── */
.toggle-outer {
    display: flex;
    justify-content: center;
    margin: 32px 0 28px 0;
}

.toggle-pill {
    display: inline-flex;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 100px;
    padding: 4px;
    gap: 4px;
    backdrop-filter: blur(16px);
}

.t-btn {
    padding: 10px 28px;
    border-radius: 100px;
    border: none;
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.01em;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
}

.t-btn-on  { background: linear-gradient(135deg,#7c5cfc,#5b3fd4); color:#fff; box-shadow: 0 4px 16px rgba(124,92,252,0.4); }
.t-btn-off { background: transparent; color: rgba(255,255,255,0.4); }

/* Streamlit toggle buttons — hidden, triggered via form */
.stForm { display: none !important; }

/* ── GLASS CARD ──────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}

.card-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.28);
    margin-bottom: 18px;
}

/* ── INPUT ───────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 18px !important;
    font-weight: 400 !important;
    letter-spacing: -0.2px !important;
    padding: 16px 20px !important;
    transition: all 0.2s ease !important;
    caret-color: #7c5cfc !important;
    outline: none !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus {
    background: rgba(124,92,252,0.08) !important;
    border-color: rgba(124,92,252,0.5) !important;
    box-shadow: 0 0 0 3px rgba(124,92,252,0.12) !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.2) !important;
}

.stTextInput label { display: none !important; }
.stTextInput > div > div { border: none !important; box-shadow: none !important; }

/* ── BUTTONS ─────────────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c5cfc, #5b3fd4) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 24px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 4px 20px rgba(124,92,252,0.3) !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(124,92,252,0.45) !important;
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── GLASS DIVIDER ───────────────────────── */
.glass-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 22px 0;
}

/* ── RESULT BOX ──────────────────────────── */
.result-glass {
    background: rgba(124,92,252,0.07);
    border: 1px solid rgba(124,92,252,0.2);
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 16px;
    position: relative;
    overflow: hidden;
}

.result-glass::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,92,252,0.6), transparent);
}

.result-text {
    font-size: 18px;
    font-weight: 400;
    color: rgba(255,255,255,0.92);
    line-height: 1.6;
    letter-spacing: -0.2px;
}

.result-meta {
    font-size: 11px;
    font-weight: 500;
    color: rgba(124,92,252,0.6);
    margin-top: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── SLIDER ──────────────────────────────── */
.stSlider { padding: 4px 0 !important; }
.stSlider label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.28) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── DOWNLOAD ────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.5) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 100px !important;
    padding: 9px 20px !important;
    width: auto !important;
    margin-top: 14px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}

.stDownloadButton > button:hover {
    border-color: rgba(124,92,252,0.4) !important;
    color: rgba(124,92,252,0.9) !important;
    background: rgba(124,92,252,0.07) !important;
    box-shadow: none !important;
}

.stSpinner > div { border-top-color: #7c5cfc !important; }

/* Hide column gaps for toggle */
.toggle-cols [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Load Model
# -----------------------------------

@st.cache_resource
def load_all():
    vocab_size    = 8978
    embedding_dim = 50
    rnn_units     = 128

    max_len = pickle.load(open("max_len_new.pkl", "rb"))

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(units=rnn_units),
        Dense(units=vocab_size, activation='softmax')
    ])
    model.build(input_shape=(None, max_len))

    weights = np.load("lstm_weights.npy", allow_pickle=True)
    model.set_weights(weights)

    tokenizer_data = pickle.load(open("tokenizer_data.pkl", "rb"))


    tokenizer = KerasTokenizer(num_words=vocab_size)
    tokenizer.word_index = tokenizer_data["word_index"]
    tokenizer.index_word = tokenizer_data["index_word"]
    tokenizer.word_counts = tokenizer_data["word_counts"]

    index_to_words = {i: w for w, i in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_words


model, tokenizer, max_len, index_to_words = load_all()


# -----------------------------------
# Helpers
# -----------------------------------

def predict_top_words(text, top_n=5):
    text = text.lower()
    seq  = tokenizer.texts_to_sequences([text])[0]
    seq  = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(seq, verbose=0)[0]
    top_indices = pred.argsort()[-top_n:][::-1]
    return [(index_to_words.get(i, ""), float(pred[i])) for i in top_indices]

def generate_sentence(seed_text, num_words):
    text = seed_text
    for _ in range(num_words):
        text += " " + predict_top_words(text, 1)[0][0]
    return text


# -----------------------------------
# Session state
# -----------------------------------

for k, v in {
    "mode": "word",
    "predicted": False,
    "suggestions": [],
    "current_text": "",
    "generated": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -----------------------------------
# Hero
# -----------------------------------

st.markdown("""
<div class="hero-wrap">
    <div style="display:flex;justify-content:center;margin-bottom:22px">
        <div class="hero-chip">
            <div class="hero-chip-dot"></div>
            LSTM · Neural Network · Shakespeare
        </div>
    </div>
    <div class="hero-title">Word<em>Flow</em></div>
    <div class="hero-sub">Type a phrase. Let the AI finish your thought.</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------
# Mode Toggle — pill using Streamlit buttons only
# -----------------------------------

mode = st.session_state.mode

w_on  = "t-btn t-btn-on"
w_off = "t-btn t-btn-off"
s_on  = "t-btn t-btn-on"
s_off = "t-btn t-btn-off"

# st.markdown(f"""
# <div class="toggle-outer">
#   <div class="toggle-pill">
#     <span class="{'t-btn t-btn-on' if mode=='word' else 't-btn t-btn-off'}"
#           style="pointer-events:none;display:inline-block;">
#       Word Predictor
#     </span>
#     <span class="{'t-btn t-btn-on' if mode=='sentence' else 't-btn t-btn-off'}"
#           style="pointer-events:none;display:inline-block;">
#       Sentence Generator
#     </span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# Invisible real buttons underneath
col1, col2 = st.columns(2)
with col1:
    if st.button("Word Predictor", key="mode_word", use_container_width=True):
        st.session_state.mode = "word"
        st.session_state.predicted = False
        st.session_state.suggestions = []
        st.rerun()
with col2:
    if st.button("Sentence Generator", key="mode_sent", use_container_width=True):
        st.session_state.mode = "sentence"
        st.session_state.generated = ""
        st.rerun()

# Hide the actual streamlit buttons visually, keep them clickable
st.markdown("""
<style>
/* Make toggle streamlit buttons transparent and overlay on top of the pill */
section.main > div > div > div > div:nth-child(5) div[data-testid="stHorizontalBlock"] {
    position: relative;
    margin-top: -58px !important;
    z-index: 10;
    opacity: 0;
    max-width: 380px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Word Predictor Mode
# -----------------------------------

if st.session_state.mode == "word":

    # st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Next Word Prediction</div>', unsafe_allow_html=True)

    user_text = st.text_input(
        "input",
        value=st.session_state.current_text,
        placeholder="Start typing a phrase...",
        key="main_input",
        label_visibility="collapsed"
    )

    if st.button("Predict Next Words", key="predict_btn"):
        if user_text.strip():
            st.session_state.current_text = user_text
            st.session_state.suggestions  = predict_top_words(user_text, top_n=5)
            st.session_state.predicted    = True

    if st.session_state.predicted and st.session_state.suggestions:
        st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Top Suggestions — click to append</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        for i, (word, prob) in enumerate(st.session_state.suggestions):
            with cols[i]:
                if st.button(f"{word}\n{prob:.0%}", key=f"pill_{i}", use_container_width=True):
                    st.session_state.current_text += " " + word
                    st.session_state.suggestions   = predict_top_words(st.session_state.current_text, top_n=5)
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------
# Sentence Generator Mode
# -----------------------------------

else:

    st.markdown('<div class="card-label">Sentence Generator</div>', unsafe_allow_html=True)

    seed_text = st.text_input(
        "seed",
        placeholder="Enter a starting phrase...",
        key="seed_input",
        label_visibility="collapsed"
    )

    num_words = st.slider("Words to generate", 1, 25, 10)

    if st.button("Generate Sentence", key="gen_btn"):
        if seed_text.strip():
            with st.spinner("Generating..."):
                st.session_state.generated = generate_sentence(seed_text, num_words)

    if st.session_state.generated:
        wc = len(st.session_state.generated.split())
        st.markdown(f"""
        <div class="result-glass">
            <div class="result-text">{st.session_state.generated}</div>
            <div class="result-meta">{wc} words generated</div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="Download",
            data=st.session_state.generated,
            file_name="wordflow.txt",
            mime="text/plain"
        )

    st.markdown('</div>', unsafe_allow_html=True)