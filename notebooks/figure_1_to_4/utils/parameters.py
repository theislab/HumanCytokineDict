from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True, slots=True)
class ReceptorExpression:
    present_threshold: float = 8.0
    high_threshold: float = 32.0
    
@dataclass(frozen=True, slots=True)
class SecondaryResponseConfig:
    # ----- pair evidence -------
    # Candidate-pair filter
    supernatant_min_log2fc: float = 0.5

    # Recipient evidence
    receptor_min: float = ReceptorExpression().present_threshold
    similarity_min: float = 0.30

    # Priority calculation
    magnitude_saturation: float = 2.0

    lineage_by_cell = {
        "B cell": "B",
        "CD4 T cell": "T", "CD8 T cell": "T", "Treg": "T", "MAIT": "T", "NKT": "T",
        "NK CD56low": "NK", "NK CD56hi": "NK",
        "CD14 Mono": "myeloid", "CD16 Mono": "myeloid", "cDC": "myeloid",
        "HSPC": "HSPC",
    }
    anchor_cell_types: tuple[str, ...] = (
        "B cell",
        "CD4 T cell",
        "NK CD56low",
        "CD14 Mono",
    )
    major_lineages: tuple[str, ...] = (
        "B",
        "T",
        "NK",
        "myeloid",
    )

    # -------- producer attribution ---------
    supernatant_min_log2fc_per_donor: float = 0.25
    warn_below_retained_donors: int = 7

    # -------- causality analysis ----------
    direct_share_min: float = 2 / 3
    indirect_share_min: float = 2 / 3

    route_overlap_min: float = 0.5
    route_min_common_donors: int = 4

    consistent_confidence: tuple[str, ...] = (
        "cross-lineage support",
        "single-lineage supported",
    )

    # plot styling
    causal_styles = {
        "immediate secondary": {"marker": "o", "label": "Plausible secondary"},
        "shared direct/indirect": {"marker": "D", "label": "Shared sec./tert."},
        "probable tertiary": {"marker": "^", "label": "Plausible tertiary"},
        "unresolved indirect": {"marker": "v", "label": "Unresolved tertiary"},
        "possibly incidental similarity": {"marker": "s", "label": "Secondary not requ."},
        "unresolved receptor coverage": {"marker": "$?$", "label": "Receptor unknown"},
        "unsupported similarity": {"marker": "_", "label": "Fully unsupported"}
    }
    white_inner_set = {"o", "^", "D", "p"}
    support_color = 'tab:green'