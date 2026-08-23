"""Cost model for Sarvam voice pipelines.

Turns measured or estimated usage into rupees, broken down by component. A rate card is
per-unit, but a call spends across STT, LLM, TTS and telephony at once, driven by turn
count, tokens per turn, characters spoken and cache hit rate.

Rates from docs.sarvam.ai (INR, 2026-08), kept in one table. Each rate carries its billing
unit, since mixing per-character and per-1K-character rates is an easy error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Unit(StrEnum):
    PER_AUDIO_HOUR = "per_audio_hour"
    PER_10K_CHARS = "per_10k_chars"
    PER_1M_TOKENS = "per_1m_tokens"
    PER_MINUTE = "per_minute"
    PER_PAGE = "per_page"


@dataclass(frozen=True)
class Rate:
    name: str
    unit: Unit
    inr: float
    note: str = ""

    def cost(self, quantity: float) -> float:
        """`quantity` is in the rate's own unit: hours, chars, tokens, minutes or pages."""
        if quantity < 0:
            raise ValueError(f"{self.name}: negative quantity {quantity}")
        divisor = {
            Unit.PER_AUDIO_HOUR: 1.0,
            Unit.PER_MINUTE: 1.0,
            Unit.PER_PAGE: 1.0,
            Unit.PER_10K_CHARS: 10_000.0,
            Unit.PER_1M_TOKENS: 1_000_000.0,
        }[self.unit]
        return self.inr * quantity / divisor


# Rate card: docs.sarvam.ai/api-reference-docs/pricing, INR, Aug 2026
RATES: dict[str, Rate] = {
    # Managed platform: blended per-minute price.
    "voice_agents_platform": Rate("Voice Agents (managed)", Unit.PER_MINUTE, 3.50,
                                  "authoring, simulation, numbers, eval, analytics included"),

    # Component pricing for self-orchestrated pipelines.
    "stt":            Rate("Speech to Text",                  Unit.PER_AUDIO_HOUR, 30.0),
    "stt_diarize":    Rate("STT + Diarization",               Unit.PER_AUDIO_HOUR, 45.0),
    "stt_translate":  Rate("STT + Translate",                 Unit.PER_AUDIO_HOUR, 30.0),

    "tts_bulbul_v2":  Rate("TTS Bulbul v2",                   Unit.PER_10K_CHARS, 15.0),
    "tts_bulbul_v3":  Rate("TTS Bulbul v3 (beta)",            Unit.PER_10K_CHARS, 30.0),

    "llm_105b_in":     Rate("Sarvam 105B input",              Unit.PER_1M_TOKENS, 29.28),
    "llm_105b_cached": Rate("Sarvam 105B cached input",       Unit.PER_1M_TOKENS, 10.98),
    "llm_105b_out":    Rate("Sarvam 105B output",             Unit.PER_1M_TOKENS, 73.20),

    "translate":      Rate("Sarvam Translate v1",             Unit.PER_10K_CHARS, 20.0),
    "transliterate":  Rate("Transliterate",                   Unit.PER_10K_CHARS, 20.0),
    "lang_id":        Rate("Language Identification",         Unit.PER_10K_CHARS, 3.50),
    "doc_digitize":   Rate("Document Digitization",           Unit.PER_PAGE, 0.50),
}

# Telephony is a carrier charge, not a Sarvam one, but it is often 20-30% of cost per call
# so omitting it distorts any build-vs-buy comparison. Typical Indian outbound SIP rate.
DEFAULT_TELEPHONY_INR_PER_MIN = 0.36

# Fully-loaded cost of a human collections agent, for comparison.
DEFAULT_HUMAN_AGENT_INR_PER_CALL = 42.0


@dataclass
class Usage:
    """Resource consumption for one conversation, measured or estimated."""

    audio_seconds: float = 0.0          # billed by STT
    tts_characters: int = 0             # billed by TTS
    llm_input_tokens: int = 0
    llm_cached_input_tokens: int = 0
    llm_output_tokens: int = 0
    telephony_seconds: float = 0.0
    turns: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            audio_seconds=self.audio_seconds + other.audio_seconds,
            tts_characters=self.tts_characters + other.tts_characters,
            llm_input_tokens=self.llm_input_tokens + other.llm_input_tokens,
            llm_cached_input_tokens=self.llm_cached_input_tokens + other.llm_cached_input_tokens,
            llm_output_tokens=self.llm_output_tokens + other.llm_output_tokens,
            telephony_seconds=self.telephony_seconds + other.telephony_seconds,
            turns=self.turns + other.turns,
        )


@dataclass
class CostBreakdown:
    """Per-component cost in rupees."""

    components: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def share(self) -> dict[str, float]:
        t = self.total
        return {k: (v / t if t else 0.0) for k, v in self.components.items()}

    def rounded(self, places: int = 4) -> dict[str, float]:
        return {k: round(v, places) for k, v in self.components.items()}


def component_cost(
    usage: Usage,
    *,
    tts_model: str = "tts_bulbul_v2",
    diarize: bool = False,
    telephony_inr_per_min: float = DEFAULT_TELEPHONY_INR_PER_MIN,
) -> CostBreakdown:
    """Cost of self-orchestrating on Sarvam's component APIs."""
    stt = RATES["stt_diarize" if diarize else "stt"]
    return CostBreakdown({
        "stt": stt.cost(usage.audio_seconds / 3600.0),
        "llm_input": RATES["llm_105b_in"].cost(usage.llm_input_tokens),
        "llm_cached": RATES["llm_105b_cached"].cost(usage.llm_cached_input_tokens),
        "llm_output": RATES["llm_105b_out"].cost(usage.llm_output_tokens),
        "tts": RATES[tts_model].cost(usage.tts_characters),
        "telephony": telephony_inr_per_min * (usage.telephony_seconds / 60.0),
    })


def platform_cost(
    usage: Usage,
    *,
    telephony_inr_per_min: float = DEFAULT_TELEPHONY_INR_PER_MIN,
) -> CostBreakdown:
    """Cost on the managed Voice Agents platform, plus carrier."""
    minutes = usage.audio_seconds / 60.0
    return CostBreakdown({
        "voice_agents_platform": RATES["voice_agents_platform"].cost(minutes),
        "telephony": telephony_inr_per_min * (usage.telephony_seconds / 60.0),
    })


def savings_vs_human(
    cost_per_call_inr: float,
    *,
    human_inr_per_call: float = DEFAULT_HUMAN_AGENT_INR_PER_CALL,
) -> dict[str, float]:
    """Savings against a human agent baseline."""
    if human_inr_per_call <= 0:
        raise ValueError("human_inr_per_call must be positive")
    saved = human_inr_per_call - cost_per_call_inr
    return {
        "ai_inr_per_call": round(cost_per_call_inr, 2),
        "human_inr_per_call": round(human_inr_per_call, 2),
        "saved_inr_per_call": round(saved, 2),
        "saving_pct": round(100.0 * saved / human_inr_per_call, 1),
    }
