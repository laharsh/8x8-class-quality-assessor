from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QualityTier(str, Enum):
    POOR = "Poor"
    AVERAGE = "Average"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class Observation(BaseModel):
    category: str = Field(description="e.g. rapport, homework, video, audio")
    detail: str
    evidence: Optional[str] = Field(
        default=None, description="Quote, timestamp, or frame note"
    )


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class VisionSummary(BaseModel):
    frames_sampled: int
    dark_frame_ratio: float
    face_visible_ratio: float
    avg_brightness: float
    notes: List[str] = Field(default_factory=list)


class AudioSummary(BaseModel):
    duration_sec: float
    mean_rms: float
    silence_ratio: float
    notes: List[str] = Field(default_factory=list)


class TranscriptFeatures(BaseModel):
    word_count: int
    has_greeting: bool
    opening_greeting: bool = False
    late_greeting: bool = False
    revision_before_warmup: bool = False
    has_conclusion: bool
    mentions_homework: bool
    mentions_video_sharing: bool = False
    individual_question_hits: int
    rapport_keywords: int
    considers_everyone: bool = False
    revision_early: bool
    new_joiner_mention: bool
    sample_quotes: List[str] = Field(default_factory=list)


class ScoreLine(BaseModel):
    label: str
    delta: int
    running_total: int


class AssessmentResult(BaseModel):
    tier: QualityTier
    confidence: str = Field(description="low | medium | high")
    observations: List[Observation]
    limitations: List[str] = Field(default_factory=list)
    method: str = Field(description="rules | rules+llm")
    rule_score: Optional[int] = None
    score_breakdown: List[ScoreLine] = Field(default_factory=list)
    transcript_features: Optional[TranscriptFeatures] = None
