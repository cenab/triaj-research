"""Advanced model architecture and custom loss for triage (ported from experimental kaggle_enhanced_final_fix_v2).

Key classes
-----------
AdvancedHierarchicalTriageModel – multi-pathway network with self-attention & residuals.
AdvancedClinicalSafetyLoss      – weighted CE + focal component + domain-specific penalties.

Both classes are self-contained and torch-script friendly.
"""

from __future__ import annotations

from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AdvancedHierarchicalTriageModel",
    "AdvancedClinicalSafetyLoss",
]


class AdvancedHierarchicalTriageModel(nn.Module):
    """Hierarchical attention model used in the Kaggle-enhanced pipeline.

    By default we replicate the feature-group sizes from the original notebook, but
    you can pass any counts you like – the attention layer adapts accordingly.
    """

    def __init__(
        self,
        num_vital_features: int = 8,
        num_symptom_features: int = 5,
        num_risk_features: int = 2,
        num_context_features: int = 4,
        num_lab_features: int = 8,
        num_interaction_features: int = 2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        # ----- pathway definitions ----------------------------
        def _mlp(in_dim: int, hidden: Sequence[int], dropout: float = 0.3):
            layers: list[nn.Module] = []
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            return nn.Sequential(*layers)

        self.vital_path = nn.Sequential(
            _mlp(num_vital_features, [128, 64], dropout=0.1), nn.Linear(64, 32)
        )
        self.symptom_path = nn.Sequential(
            _mlp(num_symptom_features, [64, 32], dropout=0.1), nn.Linear(32, 16)
        )
        self.risk_path = nn.Sequential(
            _mlp(num_risk_features, [32, 16], dropout=0.1), nn.Linear(16, 8)
        )
        self.context_path = nn.Sequential(
            _mlp(num_context_features, [32, 16], dropout=0.1), nn.Linear(16, 8)
        )
        self.lab_path = nn.Sequential(
            _mlp(num_lab_features, [32, 16], dropout=0.1), nn.Linear(16, 8)
        )
        self.interaction_path = nn.Sequential(
            _mlp(num_interaction_features, [16], dropout=0.1), nn.Linear(16, 8)
        )

        combined_dim = 32 + 16 + 8 + 8 + 8 + 8
        self.attention = nn.MultiheadAttention(
            embed_dim=combined_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.residual_transform = nn.Linear(combined_dim, combined_dim)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

        self.apply(self._init_weights)

    # ---------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------
    def forward(
        self,
        vital_data: torch.Tensor,
        symptom_data: torch.Tensor,
        risk_data: torch.Tensor,
        context_data: torch.Tensor,
        lab_data: torch.Tensor,
        interaction_data: torch.Tensor,
    ) -> torch.Tensor:
        v = self.vital_path(vital_data)
        s = self.symptom_path(symptom_data)
        r = self.risk_path(risk_data)
        c = self.context_path(context_data)
        l = self.lab_path(lab_data)
        i = self.interaction_path(interaction_data)

        combined = torch.cat([v, s, r, c, l, i], dim=1)

        attn_in = combined.unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)  # self-attn
        attn_out = attn_out.squeeze(1)

        combined = attn_out + self.residual_transform(combined)  # residual
        return self.classifier(combined)


# ---------------------------------------------------------------------------
class AdvancedClinicalSafetyLoss(nn.Module):
    """Weighted CE + focal component + clinical penalty matrix."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        critical_miss_penalty: float = 50.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None  # type: ignore
        self.critical_miss_penalty = critical_miss_penalty
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "penalty_matrix",
            torch.tensor([
                [0.0, 2.0, 5.0],  # true Green mis-pred
                [8.0, 0.0, 3.0],  # true Yellow
                [50.0, 20.0, 0.0],  # true Red
            ]),
        )

    # ---------------------------------------------------------
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        ce_loss = ce_loss_fn(outputs, targets)

        # Focal component --------------------------------------------------
        ce_raw = F.cross_entropy(outputs, targets, reduction="none")
        pt = torch.exp(-ce_raw)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_raw
        focal_loss = focal_loss.mean()

        # Penalty matrix ---------------------------------------------------
        pred_classes = outputs.argmax(dim=1)
        penalty_values = self.penalty_matrix[targets, pred_classes]
        safety_penalty = penalty_values.mean()

        # Extra miss penalty for critical cases ---------------------------
        critical_mask = targets == 2  # Red cases
        if critical_mask.any():
            critical_misses = (pred_classes[critical_mask] != 2).float().mean()
            critical_penalty = critical_misses * self.critical_miss_penalty
        else:
            critical_penalty = torch.tensor(0.0, device=outputs.device)

        total_loss = ce_loss + 0.3 * focal_loss + 0.4 * safety_penalty + 0.6 * critical_penalty
        return total_loss 