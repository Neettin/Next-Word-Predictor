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
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #f5f5f7 !important;
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
.block-container {
    max-width: 680px !important;
    padding: 80px 24px 80px !important;
    margin: 0 auto !important;
}

/* Typography */
.wf-eyebrow {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: #6e6e73;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 12px;
}

.wf-title {
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -2.5px;
    line-height: 1.0;
    color: #1d1d1f;
    text-align: center;
    margin-bottom: 10px;
}

.wf-subtitle {
    font-size: 17px;
    font-weight: 400;
    color: #6e6e73;
    text-align: center;
    line-height: 1.5;
    margin-bottom: 56px;
    letter-spacing: -0.2px;
}

/* Card */
.wf-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}

.wf-card-title {
    font-size: 13px;
    font-weight: 600;
    color: #6e6e73;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Input overrides */
.stTextInput > div > div > input {
    background: #f5f5f7 !important;
    border: none !important;
    border-radius: 12px !important;
    color: #1d1d1f !important;
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
    font-size: 19px !important;
    font-weight: 500 !important;
    letter-spacing: -0.4px !important;
    padding: 18px 20px !important;
    transition: background 0.15s ease !important;
    caret-color: #0071e3 !important;
    outline: none !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus {
    background: #ebebed !important;
    outline: none !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: #aeaeb2 !important;
    font-weight: 400 !important;
}

.stTextInput label { display: none !important; }
.stTextInput > div > div { border: none !important; box-shadow: none !important; }

/* Primary button */
div[data-testid="stButton"] > button {
    background: #0071e3 !important;
    color: #ffffff !important;
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: -0.2px !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button:hover {
    background: #0077ed !important;
}

div[data-testid="stButton"] > button:active {
    background: #006edb !important;
    transform: scale(0.98) !important;
}

/* Suggestion pills */
.pill-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}

.pill {
    background: #f5f5f7;
    border-radius: 980px;
    padding: 9px 18px;
    font-size: 15px;
    font-weight: 500;
    color: #1d1d1f;
    letter-spacing: -0.2px;
    cursor: pointer;
    transition: background 0.12s ease;
    border: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}

.pill:hover {
    background: #e8e8ed;
}

.pill-pct {
    font-size: 12px;
    font-weight: 400;
    color: #aeaeb2;
}

/* Result box */
.result-box {
    background: #f5f5f7;
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
}

.result-text {
    font-size: 19px;
    font-weight: 500;
    color: #1d1d1f;
    line-height: 1.55;
    letter-spacing: -0.4px;
}

.result-words {
    font-size: 12px;
    font-weight: 500;
    color: #aeaeb2;
    margin-top: 10px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* Slider */
.stSlider {
    padding: 0 !important;
}
.stSlider label {
    font-family: -apple-system, 'SF Pro Display', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6e6e73 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div {
    background: #d1d1d6 !important;
    height: 4px !important;
    border-radius: 2px !important;
}
.stSlider > div > div > div > div {
    background: #0071e3 !important;
    border-radius: 2px !important;
}
.stSlider > div > div > div > div > div {
    background: #ffffff !important;
    border: 2px solid #0071e3 !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
}

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #d1d1d6 !important;
    color: #0071e3 !important;
    font-family: -apple-system, sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 980px !important;
    padding: 10px 22px !important;
    width: auto !important;
    margin-top: 12px !important;
    transition: all 0.12s ease !important;
}

.stDownloadButton > button:hover {
    background: #f5f5f7 !important;
    border-color: #aeaeb2 !important;
}

/* Divider */
.wf-divider {
    height: 1px;
    background: #d1d1d6;
    margin: 32px 0;
    border-radius: 1px;
}

/* Spinner override */
.stSpinner > div {
    border-color: #0071e3 !important;
    border-right-color: transparent !important;
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

    max_len = pickle.load(open(
        r"C:\Users\nitin gupta\next_word_predictor_app\max_len_new.pkl", "rb"
    ))

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(units=rnn_units),
        Dense(units=vocab_size, activation='softmax')
    ])
    model.build(input_shape=(None, max_len))

    weights = np.load(
        r"C:\Users\nitin gupta\next_word_predictor_app\lstm_weights.npy",
        allow_pickle=True
    )
    model.set_weights(weights)

    tokenizer_data = pickle.load(open(
        r"C:\Users\nitin gupta\next_word_predictor_app\tokenizer_data.pkl", "rb"
    ))

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
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
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
    "predicted": False,
    "suggestions": [],
    "current_text": "",
    "generated": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -----------------------------------
# Header
# -----------------------------------

st.markdown('<div class="wf-eyebrow">LSTM · Neural Network</div>', unsafe_allow_html=True)
st.markdown('<div class="wf-title">WordFlow.</div>', unsafe_allow_html=True)
st.markdown('<div class="wf-subtitle">Type a phrase and let AI predict<br>what comes next.</div>', unsafe_allow_html=True)


# -----------------------------------
# Predictor Card
# -----------------------------------

st.markdown('<div class="wf-card">', unsafe_allow_html=True)
st.markdown('<div class="wf-card-title">Next Word Prediction</div>', unsafe_allow_html=True)

user_text = st.text_input(
    "input",
    value=st.session_state.current_text,
    placeholder="Things are...",
    key="main_input",
    label_visibility="collapsed"
)

predict_clicked = st.button("Predict Next Words", key="predict_btn")

if predict_clicked and user_text.strip():
    st.session_state.current_text = user_text
    st.session_state.suggestions  = predict_top_words(user_text, top_n=5)
    st.session_state.predicted     = True

if st.session_state.predicted and st.session_state.suggestions:
    st.markdown('<div class="wf-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wf-card-title">Suggestions</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (word, prob) in enumerate(st.session_state.suggestions):
        with cols[i]:
            label = f"{word}  {prob:.0%}"
            if st.button(label, key=f"pill_{i}", use_container_width=True):
                st.session_state.current_text += " " + word
                st.session_state.suggestions   = predict_top_words(st.session_state.current_text, top_n=5)
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------
# Generator Card
# -----------------------------------

st.markdown('<div class="wf-card">', unsafe_allow_html=True)
st.markdown('<div class="wf-card-title">Sentence Generator</div>', unsafe_allow_html=True)

seed_text = st.text_input(
    "seed",
    placeholder="Enter a starting phrase...",
    key="seed_input",
    label_visibility="collapsed"
)

num_words = st.slider("Number of words to generate", 1, 25, 10)

if st.button("Generate Sentence", key="gen_btn"):
    if seed_text.strip():
        with st.spinner("Generating..."):
            st.session_state.generated = generate_sentence(seed_text, num_words)

if st.session_state.generated:
    st.markdown(f"""
    <div class="result-box">
        <div class="result-text">{st.session_state.generated}</div>
        <div class="result-words">{len(st.session_state.generated.split())} words</div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label="Download",
        data=st.session_state.generated,
        file_name="wordflow.txt",
        mime="text/plain"
    )

st.markdown('</div>', unsafe_allow_html=True)