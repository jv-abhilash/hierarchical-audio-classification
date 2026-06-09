from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import config as cfg
from pipeline.inference_with_fail_safe import (
    load_model,
    predict_audio,
    preprocess_audio,
)


PROJECT_ROOT = Path(__file__).parent
DEFAULT_CHECKPOINT = cfg.CHECKPOINT_DIR / cfg.CHECKPOINT_NAME
SUBCLASS_IMAGE_DIR = PROJECT_ROOT / "assets" / "subclass_images"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@st.cache_resource
def get_model(checkpoint_path: str):
    return load_model(Path(checkpoint_path))


def find_subclass_image(subclass_name: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = SUBCLASS_IMAGE_DIR / f"{subclass_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


def confidence_color(value: float) -> str:
    if value >= 0.75:
        return "#3f7d20"
    if value >= 0.45:
        return "#b8741a"
    return "#8f3b2e"


def render_header() -> None:
    st.set_page_config(
        page_title="WildSound Inference",
        page_icon="🦉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(138, 171, 100, 0.28), transparent 30%),
                    radial-gradient(circle at top right, rgba(90, 139, 87, 0.22), transparent 28%),
                    linear-gradient(180deg, #f4f0df 0%, #e7efd9 48%, #dbe7c8 100%);
                color: #203321;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }
            .hero {
                padding: 1.4rem 1.6rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(38, 73, 36, 0.94), rgba(76, 111, 61, 0.92));
                color: #f5f3e9;
                box-shadow: 0 18px 45px rgba(47, 77, 37, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 2.2rem;
                letter-spacing: 0.02em;
            }
            .hero p {
                margin: 0;
                font-size: 1rem;
                line-height: 1.55;
                color: #edf3df;
            }
            .panel {
                background: rgba(251, 249, 240, 0.86);
                border: 1px solid rgba(76, 111, 61, 0.18);
                border-radius: 20px;
                padding: 1.1rem 1.15rem;
                box-shadow: 0 16px 35px rgba(71, 95, 56, 0.10);
            }
            .result-card {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,245,231,0.92));
                border: 1px solid rgba(76, 111, 61, 0.18);
                min-height: 120px;
            }
            .result-label {
                font-size: 0.84rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #58704b;
            }
            .result-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #223a22;
                margin-top: 0.15rem;
            }
            .result-subtext {
                color: #5e6f58;
                margin-top: 0.25rem;
                font-size: 0.95rem;
            }
            .status-pill {
                display: inline-block;
                padding: 0.4rem 0.8rem;
                border-radius: 999px;
                font-size: 0.92rem;
                font-weight: 600;
                margin-top: 0.4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>WildSound Inference Console</h1>
            <p>Drop in an audio clip and inspect its predicted main class, subclass, confidence scores, fail-safe status, and the top subclass matches in a wildlife-inspired dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_cards(result: dict) -> None:
    main_col, sub_col, status_col = st.columns(3)

    with main_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Main Class</div>
                <div class="result-value">{format_label(result["main_class"])}</div>
                <div class="result-subtext">Confidence: {result["main_confidence"]:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sub_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Subclass</div>
                <div class="result-value">{format_label(result["subclass"])}</div>
                <div class="result-subtext">Confidence: {result["sub_confidence"]:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col:
        background = "#e8f2de" if not result["fail_safe_triggered"] else "#f7e4d9"
        text_color = "#315d24" if not result["fail_safe_triggered"] else "#8a3c2d"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Inference Status</div>
                <div class="status-pill" style="background:{background}; color:{text_color};">
                    {result["status_message"].title()}
                </div>
                <div class="result-subtext">Fail-safe threshold: {result["fail_safe_threshold"]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_tables(result: dict) -> None:
    main_probs = pd.DataFrame(
        [
            {
                "Main Class": format_label(label),
                "Confidence": prob,
            }
            for label, prob in result["main_probabilities"].items()
        ]
    ).sort_values("Confidence", ascending=False)

    top_subclasses = pd.DataFrame(
        [
            {
                "Rank": item["rank"],
                "Subclass": format_label(item["subclass"]),
                "Main Class": format_label(item["main_class"]),
                "Confidence": item["confidence"],
            }
            for item in result["top_subclasses"]
        ]
    )

    left, right = st.columns((1, 1.2))
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Main Class Scores")
        styled_main = main_probs.style.format({"Confidence": "{:.2%}"}).bar(
            subset=["Confidence"],
            color="#7ca35b",
            vmin=0.0,
            vmax=1.0,
        )
        st.dataframe(styled_main, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Top Subclass Matches")
        styled_sub = top_subclasses.style.format({"Confidence": "{:.2%}"}).bar(
            subset=["Confidence"],
            color="#577d4b",
            vmin=0.0,
            vmax=1.0,
        )
        st.dataframe(styled_sub, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_subclass_image(result: dict) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Subclass Image")

    subclass_for_display = result["raw_prediction"]["subclass"]
    image_path = find_subclass_image(subclass_for_display)

    if image_path:
        st.image(str(image_path), caption=format_label(subclass_for_display), use_container_width=True)
    else:
        st.info(
            "Add a subclass image at "
            f"`assets/subclass_images/{subclass_for_display}.jpg` "
            "or `.png` to show it here."
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> tuple[Path, float, int]:
    st.sidebar.markdown("## Model Controls")
    checkpoint_input = st.sidebar.text_input(
        "Checkpoint path",
        value=str(DEFAULT_CHECKPOINT),
    )
    threshold = st.sidebar.slider(
        "Fail-safe threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.01,
    )
    top_k = st.sidebar.slider(
        "Top subclass results",
        min_value=1,
        max_value=min(8, cfg.NUM_SUBCLASSES),
        value=3,
        step=1,
    )

    st.sidebar.markdown("## Available Labels")
    st.sidebar.caption("Main classes")
    st.sidebar.write(", ".join(format_label(name) for name in cfg.MAIN_CLASSES))
    st.sidebar.caption("Subclasses")
    st.sidebar.write(", ".join(format_label(name) for name in cfg.SUBCLASSES))

    return Path(checkpoint_input), threshold, top_k


def main() -> None:
    render_header()
    checkpoint_path, threshold, top_k = render_sidebar()

    st.markdown("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Upload a clip to run hierarchical inference.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_file:
        st.caption("Waiting for an audio file.")
        return

    st.audio(uploaded_file.getvalue(), format=uploaded_file.type or "audio/wav")

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return

    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_audio_path = Path(tmp_file.name)

    try:
        with st.spinner("Running acoustic inference..."):
            model, checkpoint = get_model(str(checkpoint_path))
            spec = preprocess_audio(temp_audio_path)
            result = predict_audio(
                model,
                spec,
                fail_safe_threshold=threshold,
                top_k=top_k,
            )

        render_prediction_cards(result)

        info_col, image_col = st.columns((1.35, 0.95))
        with info_col:
            render_probability_tables(result)
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Run Details")
            st.write(
                {
                    "uploaded_file": uploaded_file.name,
                    "device": str(cfg.DEVICE),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
                    "raw_prediction": result["raw_prediction"],
                }
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with image_col:
            render_subclass_image(result)
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Confidence Snapshot")
            st.markdown(
                f"""
                <p style="margin-bottom:0.35rem;color:#4e6846;">Main class confidence</p>
                <div style="font-size:1.7rem;font-weight:700;color:{confidence_color(result["main_confidence"])};">
                    {result["main_confidence"]:.2%}
                </div>
                <p style="margin:1rem 0 0.35rem;color:#4e6846;">Subclass confidence</p>
                <div style="font-size:1.7rem;font-weight:700;color:{confidence_color(result["sub_confidence"])};">
                    {result["sub_confidence"]:.2%}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if result["fail_safe_triggered"]:
            st.warning(
                "The fail-safe was triggered. The raw best prediction is shown in run details, "
                "but the model considers this clip low-confidence."
            )
    finally:
        temp_audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
