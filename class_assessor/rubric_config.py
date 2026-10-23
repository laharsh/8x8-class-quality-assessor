"""
Tune tiers here after running with "Show scoring debug" in the app.

Compare each sample to feedbacks.pdf (e.g. B491 = Excellent).
"""

# Rule score → tier (inclusive lower bound)
TIER_MIN_SCORE = {
    "Excellent": 7,
    "Good": 4,
    "Average": 1,
    "Poor": -999,
}

# Points per signal (see assess.rule_based_assess)
WEIGHTS = {
    "greeting": 2,
    "homework": 2,
    "homework_missing_penalty": -1,
    "interaction_high": 2,  # individual_question_hits >= interaction_high_min
    "interaction_low": 1,  # >= interaction_low_min
    "interaction_high_min": 3,
    "interaction_low_min": 1,
    "revision_early_penalty": -2,
    "revision_before_warmup_penalty": -3,
    "late_greeting_penalty": -2,
    "low_rapport_penalty": -1,
    "face_visible_bonus": 1,
    "face_rare_penalty": -2,
    "dark_frames_penalty": -2,
    "low_audio_energy_penalty": -1,
    "conclusion": 1,
    "considers_everyone_bonus": 1,
    # Excellent-tier proxy when human rubric stresses 1:1 (B491-style)
    "strong_one_to_one_bonus": 2,
    "strong_one_to_one_min_questions": 15,
    # Weak-tier themes (B558, B482): new joiners + pleasant energy
    "new_joiner_flow_penalty": -2,
    "new_joiner_rapport_max": 0,
    "new_joiner_max_interaction_hits": 6,
    "flat_delivery_penalty": -2,
    "flat_delivery_rapport_max": 5,
}
