"""Martin et al. (Interspeech 2023) aspiration-probe replication.

This package is deliberately evaluation-only: it reads frozen acoustic Phase-1
checkpoints and never participates in their training.
"""

from .core import PhoneInterval, pooled_overlapping_frames, prevalence_weighted_ovo_auc

__all__ = ["PhoneInterval", "pooled_overlapping_frames", "prevalence_weighted_ovo_auc"]
