import re
from typing import List, Optional

from class_assessor.config import OPENING_GREETING_SEC
from class_assessor.models import TranscriptFeatures, TranscriptSegment

GREETING_RE = re.compile(
    r"\b(hello|hi everyone|good (morning|afternoon|evening)|welcome|namaste)\b",
    re.I,
)
CONCLUSION_RE = re.compile(
    r"\b(goodbye|see you|thank you for|wrap up|that'?s all for today|conclusion)\b",
    re.I,
)
HOMEWORK_RE = re.compile(
    r"\b(homework|home\s*work|h\.?\s*w\.?|\bhw\b|assignment|worksheet|work\s*sheet|"
    r"practice (sheet|questions|at home)|do (this|it) at home|"
    r"for next class|before next class|share (the )?h\.?w|"
    r"submit (on|by)|complete (the )?(sheet|work)|takeaway)\b",
    re.I,
)
EVERYONE_RE = re.compile(
    r"\b(everyone|every body|each (one|of you)|all of you|whole class)\b",
    re.I,
)
VIDEO_RE = re.compile(
    r"\b(share (my )?screen|screen share|play (the )?video|youtube|topic video)\b",
    re.I,
)
QUESTION_RE = re.compile(
    r"\b(what do you think|can you (tell|answer)|your turn|who (else )?wants to|"
    r"who else|yes[, ]+ [A-Z][a-z]+)\b",
    re.I,
)
RAPPORT_RE = re.compile(
    r"\b(how are you|how was your (day|week)|nice to meet|introduce yourself|tell me about yourself)\b",
    re.I,
)
NEW_JOINER_RE = re.compile(
    r"\b(new (student|joiner)|first time|introduce yourself|self[- ]?introduction)\b",
    re.I,
)
REVISION_RE = re.compile(
    r"\b(revision|recap|last (class|session)|previous (class|lesson))\b",
    re.I,
)


def _first_match_time(segments: List[TranscriptSegment], pattern: re.Pattern) -> Optional[float]:
    for s in segments:
        if pattern.search(s.text):
            return s.start
    return None


def extract_transcript_features(
    segments: List[TranscriptSegment],
) -> TranscriptFeatures:
    full = " ".join(s.text for s in segments)
    lower = full.lower()
    wc = len(full.split())

    greeting_time = _first_match_time(segments, GREETING_RE)
    revision_time = _first_match_time(segments, REVISION_RE)

    opening_greeting = (
        greeting_time is not None and greeting_time <= OPENING_GREETING_SEC
    )
    late_greeting = (
        greeting_time is not None and greeting_time > OPENING_GREETING_SEC
    )

    revision_before_warmup = False
    if revision_time is not None:
        if greeting_time is None:
            revision_before_warmup = revision_time < 300
        else:
            revision_before_warmup = revision_time < greeting_time

    first_5_min = " ".join(
        s.text for s in segments if s.end <= 300
    ).lower()
    revision_early = bool(REVISION_RE.search(first_5_min)) and not GREETING_RE.search(
        first_5_min
    )

    quotes: List[str] = []
    for s in segments[:40]:
        if HOMEWORK_RE.search(s.text) or GREETING_RE.search(s.text):
            quotes.append(
                f"[{int(s.start // 60)}:{int(s.start % 60):02d}] {s.text[:120]}"
            )
            if len(quotes) >= 4:
                break

    return TranscriptFeatures(
        word_count=wc,
        has_greeting=bool(GREETING_RE.search(lower)),
        opening_greeting=opening_greeting,
        late_greeting=late_greeting,
        revision_before_warmup=revision_before_warmup,
        has_conclusion=bool(CONCLUSION_RE.search(lower)),
        mentions_homework=bool(HOMEWORK_RE.search(lower)),
        mentions_video_sharing=bool(VIDEO_RE.search(lower)),
        individual_question_hits=len(QUESTION_RE.findall(full)),
        rapport_keywords=len(RAPPORT_RE.findall(full)),
        considers_everyone=bool(EVERYONE_RE.search(lower)),
        revision_early=revision_early,
        new_joiner_mention=bool(NEW_JOINER_RE.search(lower)),
        sample_quotes=quotes,
    )
