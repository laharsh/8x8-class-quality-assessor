# Rubric mapping (organizer → app)

## Label mapping

The challenge asks for **Poor · Average · Good · Excellent**.  
`feedbacks.pdf` uses **Weak** and **Known quality issues** — map them like this:

| feedbacks.pdf | App tier | Why |
|---------------|----------|-----|
| Excellent | **Excellent** | |
| Good | **Good** | |
| Average | **Average** | |
| Weak | **Poor** (or low Average) | Multiple structural gaps; not “disaster” but below bar |
| Known quality issues | **Poor** | Repeated / serious hygiene failures |

State this mapping in your 300-word write-up.

## Stable dimensions (not per-video IDs)

Every golden comment repeats a small set of **themes**. Production systems encode **themes**, not class IDs.

| Dimension | Examples from golden set | Automated signals (this prototype) |
|-----------|--------------------------|-------------------------------------|
| **Rapport & structure** | intro, conclusion, new joiners, revision before warm-up | opening vs late greeting, revision order, new-joiner phrases, conclusion keywords |
| **Interaction** | 1:1 questions, pleasant tone | question patterns, rapport keywords |
| **Homework** | “given HWs”, “should give HWs” | homework regex on transcript |
| **Delivery energy** | energetic, pleasant, happier | audio RMS / silence (proxy only) |
| **Video hygiene** | background, face visible, camera on, light | face ratio, dark frames (background is weak) |

You are **not** meant to add a rule for “B558”. You add or adjust weights for **dimensions** until the **golden set** mostly ranks in the right order.

## Prototype vs production

| Hackathon (now) | Production (later) |
|-----------------|---------------------|
| Regex + hand weights in `rubric_config.py` | Versioned **policy YAML** + feature store |
| Optional LLM with rubric prompt | **LLM judge** with few-shot golden examples (no per-class strings) |
| ~10 labeled videos | Thousands of labels → train **ranking model** on features |
| Calibrate by eye on debug table | **Regression tests**: each golden ID must stay in expected tier band |

Hard-coded **text** does not scale. Hard-coded **dimensions** do — human reviewers already use the same dimensions in `feedbacks.pdf`.

## How to calibrate without “cheating”

1. List dimensions each golden ID fails/passes (from human text, not from your score).  
2. Change **one weight or one feature** at a time in `rubric_config.py` / `features.py`.  
3. Re-run **all** golden samples (VTT for speed).  
4. Check **ordering** (Excellent > Good > … > Weak), not exact match on every ID.  
5. Final demo: **MP4 + fast Whisper** once per tier band.

## B558 (Weak) — what humans see vs rules

Humans: weak intro/**conclusion quality**, **no new-joiner time**, **low energy/pleasantness**.  
Rules saw: early hello + 11 question hits + conclusion keyword → **Good**.

Lesson: **presence** of hello/conclusion ≠ **quality**; need penalties for **missing new-joiner flow** and **flat delivery** even when interaction count is high.
