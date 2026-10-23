# 8x8 Class Quality Assessor

Humaniti AI 24-hour build challenge: assess children's online class recordings as **Poor · Average · Good · Excellent** with evidence-backed observations.

**Live demo:** _Add your Streamlit Cloud URL after deploy, e.g. `https://your-app.streamlit.app`_

## What it does

| Input | Processing | Output |
|-------|------------|--------|
| MP4 upload or URL (HTTP / Google Drive file link) | ffmpeg + faster-whisper (stratified or full) + OpenCV + librosa + rubric rules (+ optional LLM) | Tier + bullet observations with quotes/timestamps |

Optional **VTT transcript upload** is for **local calibration only**; judges will test with a **new MP4** (no transcript).

## Run locally

**Prerequisites:** Python 3.10+, [ffmpeg](https://ffmpeg.org/download.html) on PATH.

```powershell
cd "Humaniti AI Hackathon"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    # optional: OPENAI_API_KEY / OpenRouter
streamlit run app.py
```

Open `http://localhost:8501`.

### Environment (optional LLM)

```env
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

Restart Streamlit after editing `.env`. Use sidebar **Test LLM connection**.

## Deploy (hosted URL for judges)

**Recommended: [Streamlit Community Cloud](https://share.streamlit.io)** (free, fits the brief).

1. Push this repo to **public GitHub** (no `.env`, no sample videos).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick repo, branch `main`, main file **`app.py`**.
3. **Advanced settings → Secrets** (paste):

```toml
OPENAI_API_KEY = "sk-or-..."
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
```

4. Deploy. Linux build installs **ffmpeg** from [`packages.txt`](packages.txt).

### Hosting requirements (honest)

| Resource | Local | Streamlit Cloud free |
|----------|--------|---------------------|
| **RAM** | 4 GB+ comfortable for Whisper `tiny` | **~1 GB** — tight; use **Stratified** transcription, **tiny** Whisper |
| **CPU** | Long full-class transcribe (30–60 min video) can take 15–40 min | May **timeout** on very long full-class runs — keep **Stratified** as default |
| **Disk** | Temp video per run | Ephemeral; max upload **800 MB** (see `.streamlit/config.toml`) |
| **Cost** | Free | Free tier OK for hackathon demo |

**Rules-only mode** (LLM off) works without API secrets and is enough for tier classification.

**Alternatives:** [Hugging Face Spaces](https://huggingface.co/spaces) (Docker, more RAM on CPU basic), Railway/Render (paid tiers for heavy CPU). For this challenge, Streamlit Cloud is usually sufficient if you document stratified ASR limits.

## Pipeline

1. Ingest (upload / URL / Drive file id)  
2. Sample frames across full video (ffmpeg)  
3. Transcribe: **stratified** (opening + mid + closing clips) or **full** class  
4. Features + [`rubric_config.py`](class_assessor/rubric_config.py) scoring  
5. Optional LLM JSON refinement (OpenRouter)

See [DESIGN.md](DESIGN.md), [RUBRIC.md](RUBRIC.md), [SUBMISSION.md](SUBMISSION.md).

## Submission (Humaniti)

- Hosted link + **public GitHub** + README (this file)  
- AI chat export → [`docs/ai-development-log.md`](docs/ai-development-log.md)  
- ≤300 words (outline in [SUBMISSION.md](SUBMISSION.md))  
- **Do not** commit organizer recordings or API keys  

Dataset (study only): [Google Drive samples](https://drive.google.com/drive/folders/1V8TqCMwcei2QIphqMYxA7tpSe-Ok3aVG)

## Privacy

Recordings are processed in a temp directory and deleted after each analysis. Do not publish sample videos in the repo.

## Project layout

```
app.py                 # Streamlit UI
class_assessor/        # ingest, media, whisper, vision, assess, rubric
requirements.txt
packages.txt           # ffmpeg (Streamlit Linux)
.streamlit/config.toml
```
