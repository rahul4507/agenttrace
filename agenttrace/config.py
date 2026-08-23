"""Runtime configuration.

Read once, validated once, frozen. A missing API key fails at startup rather than partway
through a labeling run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bumping this invalidates every cached label. It is part of the cache key precisely so
# that "we changed the labeling prompt" can never silently mix old and new labels in one
# report -- which would make a version comparison meaningless.
LABELER_VERSION = "labeler-v1"


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    base_url: str = "https://api.sarvam.ai"

    # Model selection. Cheap model is the degradation target, not a different feature.
    label_model: str = "sarvam-105b"
    fallback_model: str = "sarvam-105b"

    # --- resilience knobs (all overridable; defaults chosen to be polite) ---
    request_timeout_s: float = 45.0
    connect_timeout_s: float = 10.0
    max_attempts: int = 4
    backoff_base_s: float = 0.75
    backoff_max_s: float = 20.0
    max_concurrency: int = 6           # stay well under the 60 req/min Starter limit

    circuit_fail_threshold: int = 5     # consecutive failures before we shed load
    circuit_reset_s: float = 30.0

    # --- spend guardrail ---
    run_budget_inr: float = 25.0        # of the Rs 100 free credits, cap one run at 25

    # --- paths ---
    suite_dir: Path = REPO_ROOT / "suite" / "collections"
    fixtures_dir: Path = REPO_ROOT / "fixtures"
    cache_dir: Path = REPO_ROOT / ".cache"

    offline: bool = False               # True => never touch the network

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "SARVAM_API_KEY is not set. Export it, or run with --offline to use the "
                "heuristic labeler and fixture corpus."
            )
        return self.api_key


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader.

    Deliberately does NOT overwrite an already-set environment variable: a real shell
    export must win over a checked-out file, or a developer who exported a different key
    gets silently ignored and spends an afternoon on it.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


def load_settings(**overrides) -> Settings:
    _load_dotenv(REPO_ROOT / ".env")
    env = {
        "api_key": os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY"),
        "offline": os.getenv("AGENTTRACE_OFFLINE", "").lower() in {"1", "true", "yes"},
    }
    if bud := os.getenv("AGENTTRACE_RUN_BUDGET_INR"):
        try:
            env["run_budget_inr"] = float(bud)
        except ValueError as exc:
            raise ConfigError(f"AGENTTRACE_RUN_BUDGET_INR is not a number: {bud!r}") from exc

    merged = {k: v for k, v in {**env, **overrides}.items() if v is not None}
    s = Settings(**merged)
    if s.run_budget_inr <= 0:
        raise ConfigError("run_budget_inr must be positive")
    if s.max_attempts < 1:
        raise ConfigError("max_attempts must be >= 1")
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
