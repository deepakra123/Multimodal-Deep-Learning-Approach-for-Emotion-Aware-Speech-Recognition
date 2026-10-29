from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "openai/whisper-base.en"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


st.set_page_config(
    page_title="VoxSense | Speech Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #e8edf7;
        --muted: #8d9ab3;
        --line: rgba(148, 163, 184, .16);
        --panel: rgba(17, 26, 48, .82);
        --violet: #8b5cf6;
        --cyan: #22d3ee;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(84, 55, 180, .28), transparent 30rem),
            radial-gradient(circle at 90% 15%, rgba(24, 183, 210, .16), transparent 28rem),
            #070b15;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: rgba(8, 13, 27, .88);
        border-right: 1px solid var(--line);
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.03em; }
    h1 { font-size: clamp(2.4rem, 5vw, 4.7rem) !important; line-height: 1 !important; }
    h2 { font-size: 1.65rem !important; }
    .block-container { max-width: 1240px; padding-top: 2rem; }
    .brand {
        color: #b9a2ff; font-family: 'Space Grotesk', sans-serif;
        font-size: .82rem; font-weight: 700; letter-spacing: .18em;
        text-transform: uppercase; margin-bottom: 1rem;
    }
    .hero-copy { color: var(--muted); font-size: 1.08rem; max-width: 680px; line-height: 1.7; }
    .hero-accent { color: #b9a2ff; }
    .panel, .stat, .step {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 20px; box-shadow: 0 16px 50px rgba(0,0,0,.18);
    }
    .panel { padding: 1.35rem; }
    .stat { padding: 1rem 1.15rem; min-height: 100px; }
    .stat-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; }
    .stat-value { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.35rem; font-weight: 700; margin-top: .35rem; }
    .step { padding: 1.1rem; height: 100%; }
    .step-number {
        align-items: center; background: linear-gradient(135deg, var(--violet), var(--cyan));
        border-radius: 10px; color: white; display: flex; font-weight: 700;
        height: 30px; justify-content: center; margin-bottom: .8rem; width: 30px;
    }
    .step-title { font-weight: 700; margin-bottom: .35rem; }
    .step-copy { color: var(--muted); font-size: .88rem; line-height: 1.5; }
    .eyebrow { color: var(--cyan); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .result-label { color: var(--muted); font-size: .78rem; letter-spacing: .1em; text-transform: uppercase; }
    .disclaimer { color: #7f8ba5; font-size: .78rem; line-height: 1.55; }
    .stButton > button {
        border: 1px solid rgba(139,92,246,.55); border-radius: 11px;
        background: linear-gradient(135deg, #7c3aed, #2563eb); color: white;
        font-weight: 700; min-height: 2.7rem;
    }
    .stButton > button:hover { border-color: #67e8f9; box-shadow: 0 0 24px rgba(34,211,238,.18); }
    [data-testid="stFileUploader"] {
        background: rgba(10, 17, 34, .7); border: 1px dashed rgba(139,92,246,.55);
        border-radius: 15px; padding: .55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Whisper speech model...")
def load_whisper():
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading emotion classifier...")
def load_emotion_classifier():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
    )


def transcribe(uploaded_file) -> str:
    processor, model = load_whisper()
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded_file.getbuffer())
        audio_path = Path(handle.name)

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        with torch.inference_mode():
            predicted_ids = model.generate(inputs.input_features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    finally:
        audio_path.unlink(missing_ok=True)


def classify_emotion(text: str) -> list[dict[str, float | str]]:
    results = load_emotion_classifier()(text)
    return sorted(results[0], key=lambda item: item["score"], reverse=True)


def reset_results() -> None:
    st.session_state.pop("transcript", None)
    st.session_state.pop("emotion_results", None)


st.markdown('<div class="brand">VoxSense / Speech intelligence workspace</div>', unsafe_allow_html=True)
st.title("Understand every voice.\nTurn speech into insight.")
st.markdown(
    '<p class="hero-copy">A polished multimodal workspace for transcribing speech and surfacing '
    '<span class="hero-accent">emotional signals</span> from real conversations.</p>',
    unsafe_allow_html=True,
)

stat_columns = st.columns(4)
for column, label, value in zip(
    stat_columns,
    ("Speech model", "Emotion model", "Supported audio", "Privacy"),
    ("Whisper Base", "6 signal classes", "WAV · MP3 · M4A", "Local processing"),
):
    column.markdown(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
steps = st.columns(3)
for column, number, title, copy in zip(
    steps,
    ("01", "02", "03"),
    ("Upload a voice sample", "Generate a transcript", "Read the signal"),
    (
        "Drop a short conversation or speech recording into the workspace.",
        "Whisper converts the audio into a clean, searchable transcript.",
        "A language model ranks the emotional signals in the transcript.",
    ),
):
    column.markdown(
        f'<div class="step"><div class="step-number">{number}</div>'
        f'<div class="step-title">{title}</div><div class="step-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
workspace, context = st.columns([1.45, .75], gap="large")

with context:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Demo context</div>', unsafe_allow_html=True)
    st.subheader("Built for conversations that matter")
    st.write(
        "Use this prototype to demonstrate how speech recognition can support "
        "customer experience, coaching, accessibility, and research workflows."
    )
    st.markdown(
        '<p class="disclaimer"><b>Responsible AI note:</b> emotion labels are probabilistic '
        'signals from language, not definitive judgments about a person. Always keep a '
        'human in the loop.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with workspace:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Analysis workspace</div>', unsafe_allow_html=True)
    st.subheader("Add an audio sample")
    uploaded_file = st.file_uploader(
        "Drop an audio file here",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        help="Short speech clips work best for the demo.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.audio(uploaded_file)
        action_columns = st.columns([1, 1, 2])
        with action_columns[0]:
            analyze_clicked = st.button("Analyze sample", type="primary", use_container_width=True)
        with action_columns[1]:
            if st.button("Clear", use_container_width=True):
                reset_results()
                st.rerun()

        if analyze_clicked:
            try:
                with st.spinner("Transcribing your sample..."):
                    transcript = transcribe(uploaded_file)
                if not transcript:
                    st.warning("No speech was detected in the uploaded audio.")
                else:
                    st.session_state["transcript"] = transcript
                    st.session_state.pop("emotion_results", None)
                    st.success("Analysis ready.")
            except Exception as exc:
                st.error(f"Could not analyze this audio file: {type(exc).__name__}: {exc}")
    else:
        st.caption("Recommended: a clear 10–60 second speech sample.")
    st.markdown('</div>', unsafe_allow_html=True)

transcript = st.session_state.get("transcript", "")
if transcript:
    st.write("")
    st.markdown('<div class="eyebrow">Signal report</div>', unsafe_allow_html=True)
    report, details = st.columns([1.2, .8], gap="large")
    with report:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Recognized speech</div>', unsafe_allow_html=True)
        st.text_area("Transcript", transcript, height=150, label_visibility="collapsed")
        words = len(transcript.split())
        st.caption(f"{words} words · Ready for review")
        st.markdown('</div>', unsafe_allow_html=True)

    with details:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Next step</div>', unsafe_allow_html=True)
        st.subheader("Map the emotion")
        st.write("Rank language signals to make the transcript easier to review.")
        if st.button("Run emotion analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("Reading language signals..."):
                    st.session_state["emotion_results"] = classify_emotion(transcript)
            except Exception as exc:
                st.error(f"Could not classify emotion: {type(exc).__name__}: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    results = st.session_state.get("emotion_results", [])
    if results:
        st.write("")
        columns = st.columns(min(3, len(results[:3])))
        for column, result in zip(columns, results[:3]):
            column.metric(
                str(result["label"]).title(),
                f"{float(result['score']):.1%}",
            )
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Confidence distribution</div>', unsafe_allow_html=True)
        st.bar_chart(
            {str(result["label"]).title(): float(result["score"]) for result in results},
            color="#8b5cf6",
        )
        st.markdown('</div>', unsafe_allow_html=True)
