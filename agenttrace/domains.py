"""Domain packs.

Everything customer-specific lives in `domains/<name>.yaml` plus a scenario suite in
`suite/<name>/`. The pipeline is domain-agnostic, so supporting a new vertical means adding
those two data files.

A pack carries:
  situations       taxonomy, dialogue templates for the corpus generator, keyword rules
  rule_order       keyword-labeler precedence (first match wins)
  aliases          cluster canonicalisation for this domain's synonyms
  condition_rules  complicating-factor vocabulary
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import REPO_ROOT
from .errors import ConfigError

DOMAINS_DIR = REPO_ROOT / "domains"
DEFAULT_DOMAIN = "collections"


@dataclass(frozen=True)
class DomainPack:
    name: str
    display: str
    situations: dict[str, dict]
    rule_order: list[str]
    aliases: dict[str, str]
    condition_rules: dict[str, list[str]]
    opening: str = ""
    names: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    regression: dict = field(default_factory=dict)
    products: list[str] = field(default_factory=list)

    # Views consumed by the pipeline

    @property
    def keyword_rules(self) -> list[tuple[str, list[str]]]:
        """(situation, keywords) in precedence order, skipping situations with none."""
        return [(k, self.situations[k]["keywords"])
                for k in self.rule_order
                if k in self.situations and self.situations[k].get("keywords")]

    @property
    def condition_rule_list(self) -> list[tuple[str, list[str]]]:
        return list(self.condition_rules.items())

    def in_suite(self) -> set[str]:
        """Situations the pack expects the suite to declare."""
        return {k for k, s in self.situations.items() if s.get("in_suite")}

    def suite_dir(self) -> Path:
        return REPO_ROOT / "suite" / self.name


def _validate(raw: dict, path: Path) -> None:
    for key in ("name", "situations", "rule_order"):
        if key not in raw:
            raise ConfigError(f"{path.name}: domain pack is missing required key {key!r}")
    if not isinstance(raw["situations"], dict) or not raw["situations"]:
        raise ConfigError(f"{path.name}: 'situations' must be a non-empty mapping")

    # A rule naming an unknown situation would never match and never error.
    unknown = [s for s in raw["rule_order"] if s not in raw["situations"]]
    if unknown:
        raise ConfigError(f"{path.name}: rule_order names unknown situations: {unknown}")

    for key, spec in raw["situations"].items():
        for req in ("weight", "fail", "caller", "agent"):
            if req not in spec:
                raise ConfigError(f"{path.name}: situation {key!r} is missing {req!r}")
        if not 0.0 <= spec["fail"] <= 1.0:
            raise ConfigError(f"{path.name}: situation {key!r} has fail={spec['fail']}, "
                              f"which is not a probability")


@functools.lru_cache(maxsize=8)
def load_domain(name: str = DEFAULT_DOMAIN, *, directory: Path | None = None) -> DomainPack:
    """Load and validate a domain pack. Cached; packs are immutable at runtime."""
    directory = directory or DOMAINS_DIR
    path = Path(directory) / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in Path(directory).glob("*.yaml"))
        raise ConfigError(f"no domain pack {name!r} in {directory} "
                          f"(available: {available or 'none'})")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: not valid YAML: {str(exc)[:200]}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")

    _validate(raw, path)
    return DomainPack(
        name=raw["name"],
        display=raw.get("display", raw["name"]),
        situations=raw["situations"],
        rule_order=raw["rule_order"],
        aliases=raw.get("aliases", {}) or {},
        condition_rules=raw.get("condition_rules", {}) or {},
        opening=raw.get("opening", ""),
        names=raw.get("names", []) or [],
        languages=raw.get("languages", []) or [],
        regression=raw.get("regression", {}) or {},
        products=raw.get("products", []) or [],
    )


def available_domains(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or DOMAINS_DIR).glob("*.yaml"))
