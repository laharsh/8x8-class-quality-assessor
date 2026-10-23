import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from class_assessor.env_loader import (
    get_openai_api_key,
    get_openai_base_url,
    llm_endpoint_label,
    load_project_env,
    test_llm_connection,
)
from class_assessor.pipeline import run_pipeline
from class_assessor.transcript_import import load_transcript_file

load_project_env()

st.set_page_config(
    page_title="8x8 Class Quality Assessor",
    page_icon="📚",
    layout="centered",
)

st.title("8x8 Class Quality Assessor")
st.caption(
    "Upload a recording or paste a URL. Quality tiers: Poor · Average · Good · Excellent"
)

with st.sidebar:
    st.header("Options")
    _key = get_openai_api_key()
    if _key:
        st.caption(f"API key loaded (…{_key[-4:]}) · **{llm_endpoint_label()}**")
        base = get_openai_base_url()
        if _key.startswith("sk-or-") and not base:
            st.info("OpenRouter key detected — using https://openrouter.ai/api/v1")
        if st.button("Test LLM connection"):
            try:
                reply = test_llm_connection()
                st.success(f"LLM OK: {reply}")
            except Exception as exc:
                st.error(f"LLM failed: {exc}")
    else:
        st.warning("No API key — add OPENAI_API_KEY to `.env` in project folder, then restart Streamlit.")
    use_llm = st.toggle(
        "Use LLM for richer explanations",
        value=bool(_key),
        help="`.env` must live next to `app.py`. Restart Streamlit after editing.",
        disabled=not _key,
    )
    transcribe_mode = st.radio(
        "Transcription (MP4 only)",
        options=["stratified", "full"],
        format_func=lambda x: (
            "Stratified — 3 clips: intro + mid + end (~10–15 min CPU)"
            if x == "stratified"
            else "Full class — slow (15–40+ min for long classes)"
        ),
        index=0,
    )
    show_debug = st.toggle("Show scoring debug", value=True)
    whisper_model = st.selectbox(
        "Whisper size (CPU)",
        options=["tiny", "base", "small"],
        index=0,
        help="tiny is fastest. First run downloads the model once; it stays in memory.",
    )
    st.markdown(
        "Each analysis uses a fresh temp folder (no video/transcript disk cache). "
        "Whisper weights are cached in memory after the first run."
    )

tab_upload, tab_url = st.tabs(["Upload file", "Video URL"])

uploaded = None
url = None
transcript_upload = None

with tab_upload:
    uploaded = st.file_uploader(
        "Class recording",
        type=["mp4", "webm", "mkv", "mov"],
    )
    transcript_upload = st.file_uploader(
        "Optional transcript (dev only)",
        type=["txt", "srt", "vtt", "json"],
        help="Organizer sample transcripts: speeds up calibration. Eval test is MP4-only.",
    )

with tab_url:
    url = st.text_input(
        "Direct or Google Drive link",
        placeholder="https://...",
    )

run = st.button("Analyze class", type="primary", disabled=not (uploaded or url))

if run:
    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(msg: str, frac: float) -> None:
        status.info(msg)
        progress.progress(min(1.0, max(0.0, frac)))

    work_dir = None
    try:
        if uploaded:
            suffix = "." + (uploaded.name.split(".")[-1] if "." in uploaded.name else "mp4")
            source = BytesIO(uploaded.getvalue())
        else:
            suffix = ".mp4"
            source = url.strip()

        transcript_segments = None
        if transcript_upload:
            tdir = Path(tempfile.mkdtemp(prefix="class_tx_"))
            tpath = tdir / transcript_upload.name
            tpath.write_bytes(transcript_upload.getvalue())
            transcript_segments = load_transcript_file(tpath)
            shutil.rmtree(tdir, ignore_errors=True)

        phase_log = st.empty()
        phases: list[str] = []

        def on_progress_logged(msg: str, frac: float) -> None:
            if not phases or phases[-1] != msg:
                phases.append(msg)
            on_progress(msg, frac)
            phase_log.markdown("**Pipeline log**\n\n" + "\n".join(f"- {p}" for p in phases[-8:]))

        result, work_dir = run_pipeline(
            source,
            suffix=suffix,
            use_llm=use_llm,
            progress=on_progress_logged,
            transcript_segments=transcript_segments,
            whisper_model=whisper_model,
            transcribe_mode=transcribe_mode,
        )

        tier_colors = {
            "Poor": "🔴",
            "Average": "🟠",
            "Good": "🟢",
            "Excellent": "🌟",
        }
        st.subheader(
            f"{tier_colors.get(result.tier.value, '')} **{result.tier.value}** "
            f"(confidence: {result.confidence}, via {result.method})"
        )

        st.markdown("#### Observations")
        for obs in result.observations:
            ev = f" — *{obs.evidence}*" if obs.evidence else ""
            st.markdown(f"- **{obs.category}**: {obs.detail}{ev}")

        if result.limitations:
            st.markdown("#### Limitations")
            for lim in result.limitations:
                st.markdown(f"- {lim}")

        if show_debug:
            with st.expander("Scoring debug (calibration)", expanded=True):
                st.caption(
                    "Tune `class_assessor/rubric_config.py` and regexes in `features.py`. "
                    "Target: B491 → Excellent per feedbacks.pdf."
                )
                if result.rule_score is not None:
                    st.write(f"**Rule score:** {result.rule_score}")
                if result.score_breakdown:
                    st.table(
                        [
                            {
                                "step": line.label,
                                "delta": line.delta,
                                "total": line.running_total,
                            }
                            for line in result.score_breakdown
                        ]
                    )
                if result.transcript_features:
                    st.json(result.transcript_features.model_dump())

    except Exception as e:
        st.error(f"Analysis failed: {type(e).__name__}: {e}")
        import traceback

        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.markdown(
            "Common fixes: **ffmpeg** on PATH; avoid running two ffmpeg steps at once "
            "(fixed in latest code — restart Streamlit); for LLM, use **Test LLM connection**."
        )
    finally:
        if work_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        progress.empty()
        status.empty()
