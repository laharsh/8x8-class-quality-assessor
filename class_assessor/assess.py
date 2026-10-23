import json
import os
from typing import List, Tuple

from class_assessor.env_loader import get_openai_api_key
from class_assessor.llm_chat import chat_completion_json
from class_assessor.models import (
    AssessmentResult,
    AudioSummary,
    Observation,
    QualityTier,
    ScoreLine,
    TranscriptFeatures,
    VisionSummary,
)
from class_assessor.rubric_config import TIER_MIN_SCORE, WEIGHTS
from class_assessor.transcribe import segments_to_text


def _add(score: int, breakdown: List[ScoreLine], label: str, delta: int) -> int:
    score += delta
    breakdown.append(ScoreLine(label=label, delta=delta, running_total=score))
    return score


def _tier_from_score(score: int) -> QualityTier:
    if score >= TIER_MIN_SCORE["Excellent"]:
        return QualityTier.EXCELLENT
    if score >= TIER_MIN_SCORE["Good"]:
        return QualityTier.GOOD
    if score >= TIER_MIN_SCORE["Average"]:
        return QualityTier.AVERAGE
    return QualityTier.POOR


def _rule_based_assess(
    transcript_text: str,
    tfeat: TranscriptFeatures,
    vision: VisionSummary,
    audio: AudioSummary,
) -> Tuple[AssessmentResult, int, List[ScoreLine]]:
    score = 0
    breakdown: List[ScoreLine] = []
    observations: List[Observation] = []

    if tfeat.opening_greeting:
        score = _add(score, breakdown, "opening greeting (≤3 min)", WEIGHTS["greeting"])
        observations.append(
            Observation(
                category="rapport",
                detail="Opening greeting or welcome detected early in class.",
                evidence=tfeat.sample_quotes[0] if tfeat.sample_quotes else None,
            )
        )
    elif tfeat.late_greeting:
        score = _add(
            score,
            breakdown,
            "late greeting (weak rapport start)",
            WEIGHTS["late_greeting_penalty"],
        )
        observations.append(
            Observation(
                category="rapport",
                detail="Greeting appears only after class already started — weak warm-up.",
                evidence=tfeat.sample_quotes[0] if tfeat.sample_quotes else "Transcript timing",
            )
        )
    else:
        observations.append(
            Observation(
                category="rapport",
                detail="No clear greeting in the opening.",
                evidence="Transcript scan",
            )
        )

    if tfeat.mentions_homework:
        score = _add(score, breakdown, "homework mentioned", WEIGHTS["homework"])
        observations.append(
            Observation(
                category="homework",
                detail="Homework or practice assignment mentioned.",
                evidence="Keyword match in transcript",
            )
        )
    else:
        score = _add(
            score,
            breakdown,
            "homework missing",
            WEIGHTS["homework_missing_penalty"],
        )
        observations.append(
            Observation(
                category="homework",
                detail="No homework / assignment language detected.",
                evidence="Transcript scan",
            )
        )

    if tfeat.individual_question_hits >= WEIGHTS["interaction_high_min"]:
        score = _add(score, breakdown, "strong interaction", WEIGHTS["interaction_high"])
        observations.append(
            Observation(
                category="interaction",
                detail=f"Multiple individual question patterns ({tfeat.individual_question_hits} hits).",
                evidence="Transcript patterns",
            )
        )
    elif tfeat.individual_question_hits >= WEIGHTS["interaction_low_min"]:
        score = _add(score, breakdown, "some interaction", WEIGHTS["interaction_low"])
        observations.append(
            Observation(
                category="interaction",
                detail="Some individual questioning detected.",
                evidence="Transcript patterns",
            )
        )
    else:
        observations.append(
            Observation(
                category="interaction",
                detail="Little evidence of calling on individuals.",
                evidence="Transcript patterns",
            )
        )

    if tfeat.revision_before_warmup:
        score = _add(
            score,
            breakdown,
            "revision before warm-up",
            WEIGHTS["revision_before_warmup_penalty"],
        )
        observations.append(
            Observation(
                category="rapport",
                detail="Revision/recap starts before proper greeting or rapport.",
                evidence="Transcript order",
            )
        )

    if (
        tfeat.individual_question_hits
        >= WEIGHTS["strong_one_to_one_min_questions"]
        and tfeat.opening_greeting
        and vision.face_visible_ratio >= 0.5
    ):
        score = _add(
            score,
            breakdown,
            "excellent 1:1 proxy (high questions + greeting + face)",
            WEIGHTS["strong_one_to_one_bonus"],
        )

    if tfeat.revision_early:
        score = _add(
            score,
            breakdown,
            "revision before rapport",
            WEIGHTS["revision_early_penalty"],
        )
        observations.append(
            Observation(
                category="rapport",
                detail="Revision/recap language early without a clear warm-up greeting.",
                evidence="First ~5 minutes of transcript",
            )
        )

    if not tfeat.new_joiner_mention and not tfeat.rapport_keywords:
        score = _add(
            score,
            breakdown,
            "limited rapport keywords",
            WEIGHTS["low_rapport_penalty"],
        )

    if (
        not tfeat.new_joiner_mention
        and tfeat.rapport_keywords <= WEIGHTS["new_joiner_rapport_max"]
        and tfeat.individual_question_hits
        < WEIGHTS["new_joiner_max_interaction_hits"]
    ):
        score = _add(
            score,
            breakdown,
            "no new-joiner / self-intro flow",
            WEIGHTS["new_joiner_flow_penalty"],
        )
        observations.append(
            Observation(
                category="rapport",
                detail="Little evidence of welcoming or onboarding new students.",
                evidence=f"rapport_keywords={tfeat.rapport_keywords}",
            )
        )

    if vision.face_visible_ratio >= 0.5:
        score = _add(score, breakdown, "face usually visible", WEIGHTS["face_visible_bonus"])
    elif vision.face_visible_ratio < 0.25:
        score = _add(
            score,
            breakdown,
            "face rarely visible",
            WEIGHTS["face_rare_penalty"],
        )
        observations.append(
            Observation(
                category="video",
                detail="Tutor face rarely visible in sampled frames.",
                evidence=f"{vision.face_visible_ratio:.0%} frames with face",
            )
        )

    if vision.dark_frame_ratio > 0.35:
        score = _add(
            score,
            breakdown,
            "many dark frames",
            WEIGHTS["dark_frames_penalty"],
        )
        observations.append(
            Observation(
                category="video",
                detail="Frequent dark/blank frames — possible camera off while teaching.",
                evidence=f"{vision.dark_frame_ratio:.0%} dark frames",
            )
        )

    for note in vision.notes[:2]:
        observations.append(Observation(category="video", detail=note, evidence="Frame sampling"))

    if audio.mean_rms < 0.02:
        score = _add(
            score,
            breakdown,
            "low vocal energy",
            WEIGHTS["low_audio_energy_penalty"],
        )
    if (
        audio.mean_rms < 0.02
        and tfeat.rapport_keywords <= WEIGHTS["flat_delivery_rapport_max"]
    ):
        score = _add(
            score,
            breakdown,
            "flat delivery (weak pleasant/energy proxy)",
            WEIGHTS["flat_delivery_penalty"],
        )
        observations.append(
            Observation(
                category="audio",
                detail="Low energy delivery combined with limited warm rapport language.",
                evidence="Audio RMS + rapport keywords",
            )
        )
    for note in audio.notes[:1]:
        observations.append(Observation(category="audio", detail=note, evidence="Audio RMS"))

    if tfeat.has_conclusion:
        score = _add(score, breakdown, "conclusion", WEIGHTS["conclusion"])

    if tfeat.considers_everyone:
        score = _add(
            score,
            breakdown,
            "addresses whole class",
            WEIGHTS["considers_everyone_bonus"],
        )

    tier = _tier_from_score(score)
    confidence = "medium"

    limitations = [
        "Virtual 8×8 background is not reliably detected; face visibility and lighting are approximated from sparse frames.",
        "Transcript quality depends on Whisper; children's voices may be under-represented.",
    ]

    result = AssessmentResult(
        tier=tier,
        confidence=confidence,
        observations=observations,
        limitations=limitations,
        method="rules",
        rule_score=score,
        score_breakdown=breakdown,
        transcript_features=tfeat,
    )
    return result, score, breakdown


def _llm_refine(
    rule_result: AssessmentResult,
    transcript_text: str,
    tfeat: TranscriptFeatures,
    vision: VisionSummary,
    audio: AudioSummary,
) -> AssessmentResult:
    system = """You assess live online children's class recordings for 8x8.
Tiers: Poor, Average, Good, Excellent.
Use rubric themes: rapport & intro, individual questions, homework, energetic pleasant delivery,
visible tutor face & camera on, time for new joiners, structured conclusion, virtual background when possible.
Return JSON only with keys: tier (Poor|Average|Good|Excellent), confidence (low|medium|high),
observations (list of {category, detail, evidence}), limitations (list of strings).
Every observation must cite a quote, timestamp, or metric — no generic praise."""

    user = {
        "rule_suggestion": rule_result.tier.value,
        "rule_score": rule_result.rule_score,
        "transcript_excerpt": transcript_text[:12000],
        "transcript_features": tfeat.model_dump(),
        "vision": vision.model_dump(),
        "audio": audio.model_dump(),
    }

    raw = chat_completion_json(system, json.dumps(user))
    tier = QualityTier(raw.get("tier", rule_result.tier.value))
    obs = [Observation(**o) for o in raw.get("observations", [])]
    if not obs:
        obs = rule_result.observations
    return AssessmentResult(
        tier=tier,
        confidence=raw.get("confidence", rule_result.confidence),
        observations=obs,
        limitations=raw.get("limitations", rule_result.limitations),
        method="rules+llm",
        rule_score=rule_result.rule_score,
        score_breakdown=rule_result.score_breakdown,
        transcript_features=rule_result.transcript_features,
    )


def assess(
    segments,
    tfeat: TranscriptFeatures,
    vision: VisionSummary,
    audio: AudioSummary,
    *,
    use_llm: bool = True,
) -> AssessmentResult:
    transcript_text = segments_to_text(segments)
    rule_result, _, _ = _rule_based_assess(transcript_text, tfeat, vision, audio)

    api_key = get_openai_api_key()
    if use_llm and api_key:
        try:
            return _llm_refine(rule_result, transcript_text, tfeat, vision, audio)
        except Exception as exc:
            rule_result.limitations.append(
                f"LLM refinement failed ({type(exc).__name__}: {exc}); rule-based only."
            )
    return rule_result
