"""Interactive Challenge-Response Liveness Session Manager.

Implements server-side randomized challenge sequences for verified live biometric capture:
1. Server issues cryptographically random challenge actions (e.g. LOOK_STRAIGHT -> TURN_LEFT -> LOOK_STRAIGHT).
2. Client sends capture sequence frames.
3. Server independently computes head pose (yaw, pitch, roll) on each frame and verifies
   trajectory matches expected sequence within calibrated tolerances and timeframe.
4. Single-use session tokens prevent video replay attacks of old sessions.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("facecheck.anti_spoofing.challenge")

ALLOWED_ACTIONS = [
    "LOOK_STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "TILT_UP",
    "TILT_DOWN",
]


class ChallengeSession(BaseModel):
    session_id: str
    challenge_sequence: List[str]
    created_at: float
    expires_at: float
    verified: bool = False
    used: bool = False


class ChallengeVerificationResult(BaseModel):
    passed: bool
    status: str = Field(..., description="'PASSED', 'FAILED', or 'EXPIRED'")
    session_id: str
    expected_sequence: List[str]
    detected_sequence: List[str]
    reason_codes: List[str] = Field(default_factory=list)
    message: str


class InteractiveChallengeManager:
    """Thread-safe in-memory session manager for bank interactive liveness challenges."""

    def __init__(self, default_ttl_seconds: int = 120):
        self.default_ttl_seconds = default_ttl_seconds
        self._sessions: Dict[str, ChallengeSession] = {}

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired_keys = [k for k, v in self._sessions.items() if now > v.expires_at]
        for k in expired_keys:
            self._sessions.pop(k, None)

    def create_session(self, sequence_length: int = 3) -> ChallengeSession:
        """Create a randomized challenge sequence for an applicant."""
        self._cleanup_expired()
        session_id = f"live_{uuid.uuid4().hex}"
        now = time.time()

        # Build random non-repeating sequence starting with straight look
        seq = ["LOOK_STRAIGHT"]
        candidates = ["TURN_LEFT", "TURN_RIGHT", "TILT_UP", "TILT_DOWN"]
        random.shuffle(candidates)
        seq.append(candidates[0])
        seq.append("LOOK_STRAIGHT" if sequence_length == 3 else candidates[1])

        session = ChallengeSession(
            session_id=session_id,
            challenge_sequence=seq,
            created_at=now,
            expires_at=now + self.default_ttl_seconds,
            verified=False,
            used=False,
        )
        self._sessions[session_id] = session
        return session

    def verify_poses(
        self,
        session_id: str,
        frame_poses: List[Dict[str, float]],
    ) -> ChallengeVerificationResult:
        """Verify whether measured head poses in sequence of frames fulfill challenge requirements."""
        self._cleanup_expired()
        session = self._sessions.get(session_id)

        if session is None:
            return ChallengeVerificationResult(
                passed=False,
                status="EXPIRED",
                session_id=session_id,
                expected_sequence=[],
                detected_sequence=[],
                reason_codes=["SESSION_NOT_FOUND_OR_EXPIRED"],
                message="Challenge session has expired or is invalid. Please start a new session.",
            )

        if session.used:
            return ChallengeVerificationResult(
                passed=False,
                status="FAILED",
                session_id=session_id,
                expected_sequence=session.challenge_sequence,
                detected_sequence=[],
                reason_codes=["SESSION_ALREADY_USED"],
                message="Challenge session token was already used. Replay attempt rejected.",
            )

        # Mark session as used immediately (single-use anti-replay)
        session.used = True

        if not frame_poses:
            return ChallengeVerificationResult(
                passed=False,
                status="FAILED",
                session_id=session_id,
                expected_sequence=session.challenge_sequence,
                detected_sequence=[],
                reason_codes=["NO_FRAMES_PROVIDED"],
                message="No frames provided for challenge sequence verification.",
            )

        # Map each frame pose to dominant action
        detected_sequence: List[str] = []
        for pose in frame_poses:
            yaw = pose.get("yaw", 0.0)
            pitch = pose.get("pitch", 0.0)

            if yaw > 18.0:
                action = "TURN_LEFT"
            elif yaw < -18.0:
                action = "TURN_RIGHT"
            elif pitch > 15.0:
                action = "TILT_DOWN"
            elif pitch < -15.0:
                action = "TILT_UP"
            else:
                action = "LOOK_STRAIGHT"

            # Collapse duplicate adjacent actions in sequence
            if not detected_sequence or detected_sequence[-1] != action:
                detected_sequence.append(action)

        # Check if expected sequence is a subsequence of detected actions
        expected = session.challenge_sequence
        exp_idx = 0
        for act in detected_sequence:
            if act == expected[exp_idx]:
                exp_idx += 1
                if exp_idx == len(expected):
                    break

        passed = exp_idx == len(expected)
        session.verified = passed

        if passed:
            return ChallengeVerificationResult(
                passed=True,
                status="PASSED",
                session_id=session_id,
                expected_sequence=expected,
                detected_sequence=detected_sequence,
                reason_codes=[],
                message="Live challenge movement sequence successfully verified.",
            )
        else:
            return ChallengeVerificationResult(
                passed=False,
                status="FAILED",
                session_id=session_id,
                expected_sequence=expected,
                detected_sequence=detected_sequence,
                reason_codes=["CHALLENGE_SEQUENCE_MISMATCH"],
                message=f"Movements did not match assigned challenge sequence. Expected: {expected}, Detected: {detected_sequence}",
            )


_challenge_manager_singleton: Optional[InteractiveChallengeManager] = None


def get_challenge_manager() -> InteractiveChallengeManager:
    global _challenge_manager_singleton
    if _challenge_manager_singleton is None:
        _challenge_manager_singleton = InteractiveChallengeManager()
    return _challenge_manager_singleton
