# Class Quality Assessor — Design Plan

## Goal

Hosted prototype that accepts a **class recording** (file upload or URL), runs an automated multimodal pipeline, and returns **Poor / Average / Good / Excellent** with **specific, evidence-backed observations** aligned with 8x8’s human feedback rubric.

## Constraints (from challenge)

- ~4–5 hours build target; working demo > perfect ML
- Confidential recordings — ephemeral processing, no videos in git
- Unseen test recording at evaluation time
- Submit: hosted link, GitHub, AI logs, ≤300 word write-up

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────────┐
│ Upload/URL  │────▶│ Ingest       │────▶│ Local temp video file           │
└─────────────┘     └──────────────┘     └───────────────┬─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┼───────────────────────────────┐
         ▼                       ▼                       ▼                               ▼
   ┌───────────┐          ┌────────────┐          ┌─────────────┐                 ┌────────────┐
   │ ffmpeg    │          │ ffmpeg     │          │ OpenCV      │                 │ (optional) │
   │ full audio│          │ frame      │          │ face/light  │                 │ LLM assess │
   └─────┬─────┘          │ samples    │          │ heuristics  │                 └─────┬──────┘
         │                └─────┬──────┘          └──────┬──────┘                       │
         ▼                      │                        │                              │
   ┌───────────┐                 └──────────┬─────────────┘                              │
   │ Whisper   │                            ▼                                            │
   │ transcript│                    ┌───────────────┐                                      │
   └─────┬─────┘                    │ Signal bundle │◀─────────────────────────────────────┘
         │                          └───────┬───────┘
         └──────────────────┬─────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Rubric scorer    │  rules + optional OpenAI structured JSON
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Streamlit UI     │
                    └──────────────────┘
```

## Signal layers (mapped to `feedbacks.pdf`)

| Signal | Source | Rubric themes |
|--------|--------|----------------|
| Full transcript + timestamps | Whisper on extracted audio | Intro, conclusion, HW, individual questions, rapport vs jump to revision, new-joiner time |
| Frame samples (every N sec) | ffmpeg | 8×8 virtual background (proxy: blur/uniform backdrop heuristics + vision LLM if enabled) |
| Face visibility / camera off | OpenCV Haar + dark-frame ratio | Face visible, camera on while teaching |
| Audio dynamics | librosa RMS / silence ratio | Energetic vs flat delivery |
| Lexical patterns | regex + transcript scan | Homework, greeting, names/questions |

## Scoring strategy

1. **Rule layer** (always on): checklist scores from transcript + vision heuristics → suggested tier + bullet evidence.
2. **LLM layer** (if `OPENAI_API_KEY`): multimodal or text-only refinement with strict JSON schema; must cite timestamps/quotes.
3. **Merge**: LLM can adjust tier only when evidence list is non-empty; otherwise fall back to rules.

No fine-tuning in v1 — calibration via rubric weights tuned against sample IDs in `feedbacks.pdf`.

## Ingest

- **Upload**: Streamlit `file_uploader` → temp file
- **URL**: HTTP(S) direct download; Google Drive folder/file via `gdown` when link matches Drive patterns
- Max duration guard (e.g. 90 min) and size limit configurable

## Deployment

- **Local**: `streamlit run app.py`
- **Hosted**: Streamlit Community Cloud — set secrets for API key; ensure `ffmpeg` available (apt in `packages.txt` for Cloud)

## Security & privacy

- Temp dir per request; delete on completion
- `.gitignore` for `data/`, `.env`, `*.mp4`
- README warns against committing recordings

## Milestones

- [x] Design doc + repo scaffold
- [ ] Ingest (upload + URL + Drive)
- [ ] Media extraction + Whisper
- [ ] Vision heuristics + transcript features
- [ ] Assessment + Streamlit UI
- [ ] README + run instructions

## Future (if time)

- Finetune small frame classifier on auto-labeled frames from sample videos
- Speaker diarization for teacher vs students
- Adaptive frame sampling when scene change detected
