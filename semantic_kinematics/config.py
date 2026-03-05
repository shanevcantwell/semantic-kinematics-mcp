"""
Configuration loader and validator for the semantic kinematics pipeline.

Loads config.yaml from repo root and provides typed access to settings.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SourcesConfig:
    """Source file paths."""
    html_takeout: Optional[Path] = None
    json_takeout: Optional[Path] = None
    webui_dir: Optional[Path] = None


@dataclass
class StagesConfig:
    """Stage output directories."""
    a1_historical: Path = Path("data/stage_a1_historical/")
    b1_webui: Path = Path("data/stage_b1_webui/")
    b2_recent: Path = Path("data/stage_b2_recent/")
    b3_enriched: Path = Path("data/stage_b3_enriched/")
    c1_unified: Path = Path("data/stage_c1_unified/")


@dataclass
class OutputConfig:
    """Final output paths."""
    timeline: Path = Path("data/embedding_timeline.json")
    html: Path = Path("data/timeline.html")


@dataclass
class MatchingConfig:
    """B3 matching parameters."""
    jaccard_threshold: float = 0.8
    length_ratio_min: float = 0.7


@dataclass
class ProvenanceConfig:
    """Provenance tracking settings."""
    preserve_html: bool = True
    extraction_version: str = "1.0"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Build config from dictionary."""
        config = cls()

        # Sources
        if 'sources' in data:
            src = data['sources']
            if src.get('html_takeout'):
                config.sources.html_takeout = Path(src['html_takeout'])
            if src.get('json_takeout'):
                config.sources.json_takeout = Path(src['json_takeout'])
            if src.get('webui_dir'):
                config.sources.webui_dir = Path(src['webui_dir'])

        # Stages
        if 'stages' in data:
            stg = data['stages']
            for key in ['a1_historical', 'b1_webui', 'b2_recent', 'b3_enriched', 'c1_unified']:
                if stg.get(key):
                    setattr(config.stages, key, Path(stg[key]))

        # Output
        if 'output' in data:
            out = data['output']
            if out.get('timeline'):
                config.output.timeline = Path(out['timeline'])
            if out.get('html'):
                config.output.html = Path(out['html'])

        # Matching
        if 'matching' in data:
            m = data['matching']
            if 'jaccard_threshold' in m:
                config.matching.jaccard_threshold = float(m['jaccard_threshold'])
            if 'length_ratio_min' in m:
                config.matching.length_ratio_min = float(m['length_ratio_min'])

        # Provenance
        if 'provenance' in data:
            p = data['provenance']
            if 'preserve_html' in p:
                config.provenance.preserve_html = bool(p['preserve_html'])
            if 'extraction_version' in p:
                config.provenance.extraction_version = str(p['extraction_version'])

        return config

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors (empty if valid)."""
        errors = []

        # Check source files exist
        if self.sources.html_takeout and not self.sources.html_takeout.exists():
            errors.append(f"HTML takeout not found: {self.sources.html_takeout}")

        if self.sources.json_takeout and not self.sources.json_takeout.exists():
            errors.append(f"JSON takeout not found: {self.sources.json_takeout}")

        if self.sources.webui_dir and not self.sources.webui_dir.exists():
            errors.append(f"Web UI directory not found: {self.sources.webui_dir}")

        # Check matching thresholds are valid
        if not 0 <= self.matching.jaccard_threshold <= 1:
            errors.append(f"jaccard_threshold must be 0-1, got: {self.matching.jaccard_threshold}")

        if not 0 <= self.matching.length_ratio_min <= 1:
            errors.append(f"length_ratio_min must be 0-1, got: {self.matching.length_ratio_min}")

        return errors

    def ensure_stage_dirs(self):
        """Create stage output directories if they don't exist."""
        for stage_dir in [
            self.stages.a1_historical,
            self.stages.b1_webui,
            self.stages.b2_recent,
            self.stages.b3_enriched,
            self.stages.c1_unified,
        ]:
            stage_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """
    Load pipeline configuration.

    Args:
        config_path: Path to config.yaml. If None, looks in repo root.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config validation fails
    """
    if config_path is None:
        # Look for config.yaml in repo root (parent of semantic_kinematics/)
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = PipelineConfig.from_yaml(config_path)

    errors = config.validate()
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
