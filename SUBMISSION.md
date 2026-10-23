# Humaniti AI — submission checklist

From **Humaniti_AI_24_Hour_Build_Challenge.pdf**:

| Deliverable | Status | Notes |
|-------------|--------|--------|
| **Hosted app URL** | ☐ | Streamlit Cloud / HF Spaces — judges upload or paste URL |
| **GitHub repo** | ☐ | Public, no secrets, no sample videos |
| **README** | ☐ | How to run locally + deploy |
| **AI interaction history** | ☐ | Export Cursor chat / redact keys — link or `docs/ai-logs/` |
| **≤300 word write-up** | ☐ | Approach, what worked/failed, limits — paste in Google Form |

**Form:** https://docs.google.com/forms/d/e/1FAIpQLSemgdFdy8ijCNhKkkjhukNh71MCkfTEs2ho-3NWPEfc0j9KyQ/viewform

## Suggested 300-word outline

1. **Pipeline:** URL/upload → ffmpeg frames + stratified Whisper (or VTT dev) → rubric features → tier + evidence.  
2. **Why multimodal:** 8x8 feedback stresses rapport, HW, 1:1 questions, energy, face/camera — not transcript alone.  
3. **Calibration:** `feedbacks.pdf` golden set; Weak/Known issues → Poor; rules in `rubric_config.py`.  
4. **What worked:** Relative ranking on samples; Google Drive ingest; optional OpenRouter LLM.  
5. **Limits:** Stratified ASR misses mid-class; no true virtual-background detector; CPU Whisper slow on full 60 min classes.  
6. **Privacy:** Ephemeral temp files; no recordings in repo.

## AI logs

1. In Cursor: export or copy this chat thread (redact API keys).  
2. Save as `docs/ai-development-log.md` or PDF zip for the form.  
3. Do **not** commit organizer class videos.
