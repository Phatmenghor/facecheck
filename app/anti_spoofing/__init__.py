"""Presentation-Attack Detection (PAD) and Liveness package."""

from app.anti_spoofing.texture_detector import detect_print_attack
from app.anti_spoofing.replay_detector import detect_screen_replay
from app.anti_spoofing.challenge import (
    ChallengeSession,
    ChallengeVerificationResult,
    InteractiveChallengeManager,
    get_challenge_manager,
)
from app.anti_spoofing.pad_service import PADResult, AttackSignal, evaluate_pad

__all__ = [
    "detect_print_attack",
    "detect_screen_replay",
    "ChallengeSession",
    "ChallengeVerificationResult",
    "InteractiveChallengeManager",
    "get_challenge_manager",
    "PADResult",
    "AttackSignal",
    "evaluate_pad",
]
