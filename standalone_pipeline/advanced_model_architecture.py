"""Advanced model architecture and custom loss for triage (ported from experimental kaggle_enhanced_final_fix_v2).

Key classes
-----------
AdvancedHierarchicalTriageModel – multi-pathway network with self-attention & residuals.
AdvancedHierarchicalTriageEnsemble - Hierarchical ensemble with gating + specialist heads.
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
    "AdvancedHierarchicalTriageEnsemble",
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
        num_classes: int = 5,
        calibration_temperature: float = 1.0,
        class_thresholds: Optional[Sequence[float]] = None,
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
            _mlp(num_risk_features, [32, 16], dropout=0.1), nn.Linear(16, 16)
        )
        self.context_path = nn.Sequential(
            _mlp(num_context_features, [32, 16], dropout=0.1), nn.Linear(16, 16)
        )
        self.lab_path = nn.Sequential(
            _mlp(num_lab_features, [32, 16], dropout=0.1), nn.Linear(16, 32)
        )
        self.interaction_path = nn.Sequential(
            _mlp(num_interaction_features, [16], dropout=0.1), nn.Linear(16, 16)
        )

        self.token_dim = 64
        self.vital_proj = nn.Linear(32, self.token_dim)
        self.symptom_proj = nn.Linear(16, self.token_dim)
        self.risk_proj = nn.Linear(16, self.token_dim)
        self.context_proj = nn.Linear(16, self.token_dim)
        self.lab_proj = nn.Linear(32, self.token_dim)
        self.interaction_proj = nn.Linear(16, self.token_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.token_residual = nn.Linear(self.token_dim, self.token_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.token_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, num_classes),
        )

        if calibration_temperature <= 0:
            raise ValueError("calibration_temperature must be positive.")
        self.register_buffer(
            "calibration_temperature",
            torch.tensor(float(calibration_temperature), dtype=torch.float32),
        )

        if class_thresholds is not None and len(class_thresholds) != num_classes:
            raise ValueError(
                "class_thresholds length must match num_classes when provided."
            )
        threshold_tensor = (
            torch.tensor(class_thresholds, dtype=torch.float32)
            if class_thresholds is not None
            else torch.zeros(num_classes, dtype=torch.float32)
        )
        self.register_buffer("class_thresholds", threshold_tensor)
        self._thresholds_enabled = class_thresholds is not None

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
        v = self.vital_proj(self.vital_path(vital_data))
        s = self.symptom_proj(self.symptom_path(symptom_data))
        r = self.risk_proj(self.risk_path(risk_data))
        c = self.context_proj(self.context_path(context_data))
        l = self.lab_proj(self.lab_path(lab_data))
        i = self.interaction_proj(self.interaction_path(interaction_data))

        tokens = torch.stack([v, s, r, c, l, i], dim=1)  # (B, 6, token_dim)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        pooled = attn_out.mean(dim=1)
        combined = pooled + self.token_residual(tokens.mean(dim=1))
        return self.classifier(combined)

    # ---------------------------------------------------------
    def _compute_probs(
        self, logits: torch.Tensor, temperature: Optional[float] = None
    ) -> torch.Tensor:
        temp_tensor: torch.Tensor
        if temperature is None:
            temp_tensor = self.calibration_temperature.to(logits.device, logits.dtype)
        else:
            if temperature <= 0:
                raise ValueError("temperature must be positive.")
            temp_tensor = torch.tensor(
                float(temperature), dtype=logits.dtype, device=logits.device
            )
        calibrated_logits = logits / temp_tensor.clamp_min(1e-6)
        return F.softmax(calibrated_logits, dim=-1)

    # ---------------------------------------------------------
    def predict_proba(
        self,
        vital_data: torch.Tensor,
        symptom_data: torch.Tensor,
        risk_data: torch.Tensor,
        context_data: torch.Tensor,
        lab_data: torch.Tensor,
        interaction_data: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        logits = self.forward(
            vital_data,
            symptom_data,
            risk_data,
            context_data,
            lab_data,
            interaction_data,
        )
        return self._compute_probs(logits, temperature)

    # ---------------------------------------------------------
    def predict(
        self,
        vital_data: torch.Tensor,
        symptom_data: torch.Tensor,
        risk_data: torch.Tensor,
        context_data: torch.Tensor,
        lab_data: torch.Tensor,
        interaction_data: torch.Tensor,
        *,
        temperature: Optional[float] = None,
        thresholds: Optional[Sequence[float]] = None,
        return_probs: bool = False,
    ):
        probs = self.predict_proba(
            vital_data,
            symptom_data,
            risk_data,
            context_data,
            lab_data,
            interaction_data,
            temperature=temperature,
        )

        thresholds_tensor: Optional[torch.Tensor]
        if thresholds is not None:
            if len(thresholds) != probs.size(-1):
                raise ValueError("thresholds length must match number of classes.")
            thresholds_tensor = torch.tensor(
                thresholds, dtype=probs.dtype, device=probs.device
            )
        elif self._thresholds_enabled:
            thresholds_tensor = self.class_thresholds.to(probs.device, probs.dtype)
        else:
            thresholds_tensor = None

        if thresholds_tensor is not None:
            adjusted = probs - thresholds_tensor
            best_vals, preds = adjusted.max(dim=-1)
            fallback = probs.argmax(dim=-1)
            preds = torch.where(best_vals < 0, fallback, preds)
        else:
            preds = probs.argmax(dim=-1)

        return (preds, probs) if return_probs else preds

    # ---------------------------------------------------------
    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.calibration_temperature.fill_(float(temperature))

    # ---------------------------------------------------------
    def get_temperature(self) -> float:
        return float(self.calibration_temperature.item())

    # ---------------------------------------------------------
    def set_class_thresholds(
        self, thresholds: Sequence[float], *, enable: bool = True
    ) -> None:
        if len(thresholds) != self.class_thresholds.numel():
            raise ValueError("thresholds length must match number of classes.")
        new_thresholds = torch.tensor(
            thresholds,
            dtype=self.class_thresholds.dtype,
            device=self.class_thresholds.device,
        )
        self.class_thresholds.copy_(new_thresholds)
        self._thresholds_enabled = bool(enable)

    # ---------------------------------------------------------
    def enable_class_thresholds(self, enabled: bool) -> None:
        self._thresholds_enabled = bool(enabled)

    # ---------------------------------------------------------
    def get_class_thresholds(self) -> list[float]:
        return self.class_thresholds.detach().cpu().tolist()


class AdvancedHierarchicalTriageEnsemble(nn.Module):
    """Hierarchical ensemble with gating + specialist heads.

    - Gate head learns Critical vs Non-Critical (binary).
    - Non-critical head classifies within the non-critical band.
    - Optional Critical-detail head classifies within the critical band.

    This generalises the legacy 3-class design (Green/Yellow/Red) and supports
    5-class ESI by setting noncritical_classes=3 (ESI5/ESI4/ESI3) and
    critical_classes=2 (ESI2/ESI1).
    """

    def __init__(
        self,
        num_vital_features: int = 8,
        num_symptom_features: int = 5,
        num_risk_features: int = 2,
        num_context_features: int = 4,
        num_lab_features: int = 8,
        num_interaction_features: int = 2,
        latent_dim: int = 32,
        noncritical_classes: int = 2,
        critical_classes: int = 1,
    ) -> None:
        super().__init__()

        self.backbone = AdvancedHierarchicalTriageModel(
            num_vital_features=num_vital_features,
            num_symptom_features=num_symptom_features,
            num_risk_features=num_risk_features,
            num_context_features=num_context_features,
            num_lab_features=num_lab_features,
            num_interaction_features=num_interaction_features,
            num_classes=latent_dim,
        )

        self.register_buffer(
            "calibration_temperature", torch.tensor(1.0, dtype=torch.float32)
        )

        # Gate: Non-critical vs Critical
        self.red_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 2),  # logits: [Non-Critical, Critical]
        )
        # Non-critical specialist head
        self.noncritical_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, noncritical_classes),
        )
        # Optional critical-detail head (e.g., ESI2 vs ESI1)
        self.critical_head = (
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, critical_classes),
            )
            if critical_classes and critical_classes > 1
            else None
        )
        self._noncritical_classes = int(noncritical_classes)
        self._critical_classes = int(critical_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, *features: torch.Tensor):
        latent = self.backbone(
            *features
        )  # (B, latent_dim)
        gate_logits = self.red_head(latent)
        noncrit_logits = self.noncritical_head(latent)
        if self.critical_head is not None:
            crit_logits = self.critical_head(latent)
            return gate_logits, noncrit_logits, crit_logits
        return gate_logits, noncrit_logits

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.calibration_temperature.fill_(float(temperature))


# ---------------------------------------------------------------------------
class AdvancedClinicalSafetyLoss(nn.Module):
    """Weighted CE + focal component + clinical penalty matrix."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        critical_miss_penalty: float = 50.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        # term weights (allow baseline to be accuracy-focused)
        w_focal: float = 0.3,
        w_safety: float = 0.4,
        w_critical: float = 0.6,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None  # type: ignore
        self.critical_miss_penalty = critical_miss_penalty
        self.alpha = alpha
        self.gamma = gamma
        self.w_focal = float(w_focal)
        self.w_safety = float(w_safety)
        self.w_critical = float(w_critical)

    def _build_penalty_matrix(self, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Base penalties: under-triage grows quadratically with distance; over-triage linearly
        mat = torch.zeros((k, k), dtype=dtype, device=device)
        for i in range(k):
            for j in range(k):
                if j < i:
                    dist = i - j
                    mat[i, j] = (dist ** 2) * 5.0
                elif j > i:
                    dist = j - i
                    mat[i, j] = dist * 1.0
                else:
                    mat[i, j] = 0.0
        return mat

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
        k = outputs.size(-1)
        penalty_matrix = self._build_penalty_matrix(k, outputs.device, outputs.dtype)
        penalty_values = penalty_matrix[targets, pred_classes]
        safety_penalty = penalty_values.mean()

        # Extra miss penalty for critical cases (dynamic) -----------------
        if k == 3:
            critical_threshold = 2
        else:
            critical_threshold = max(1, k - 2)
        critical_mask = targets >= critical_threshold
        if critical_mask.any():
            critical_misses = (pred_classes[critical_mask] < critical_threshold).float().mean()
            critical_penalty = critical_misses * self.critical_miss_penalty
        else:
            critical_penalty = torch.tensor(0.0, device=outputs.device)

        total_loss = (
            ce_loss
            + self.w_focal * focal_loss
            + self.w_safety * safety_penalty
            + self.w_critical * critical_penalty
        )
        return total_loss 

    # ---------------------------------------------------------
    def compute_hierarchical_loss(
        self, red_logits: torch.Tensor, yg_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        device = targets.device
        red_probs = torch.softmax(red_logits, dim=1)
        yg_probs = torch.softmax(yg_logits, dim=1)

        noncrit_gate = red_probs[:, 0:1]
        crit_gate = red_probs[:, 1:2]
        nc0 = noncrit_gate * yg_probs[:, 0:1]
        nc1 = noncrit_gate * yg_probs[:, 1:2]
        probs = torch.cat([nc0, nc1, crit_gate], dim=1).clamp_min(1e-8)
        log_probs = torch.log(probs)

        ce_loss = torch.nn.functional.nll_loss(log_probs, targets, weight=self.class_weights, reduction="mean")

        ce_raw = torch.nn.functional.nll_loss(log_probs, targets, reduction="none")
        pt = torch.exp(-ce_raw)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_raw
        focal_loss = focal_loss.mean()

        pred_classes = probs.argmax(dim=1)
        k = probs.size(-1)
        penalty_matrix = self._build_penalty_matrix(k, probs.device, probs.dtype)
        penalty_values = penalty_matrix[targets, pred_classes]
        safety_penalty = penalty_values.mean()

        if k == 3:
            critical_threshold = 2
        else:
            critical_threshold = max(1, k - 2)
        critical_mask = targets >= critical_threshold
        if critical_mask.any():
            critical_misses = (pred_classes[critical_mask] < critical_threshold).float().mean()
            critical_penalty = critical_misses * self.critical_miss_penalty
        else:
            critical_penalty = torch.tensor(0.0, device=device)

        total_loss = (
            ce_loss
            + self.w_focal * focal_loss
            + self.w_safety * safety_penalty
            + self.w_critical * critical_penalty
        )
        return total_loss

    # ---------------------------------------------------------
    def compose_hierarchical_probs(self, outputs: tuple) -> torch.Tensor:
        """Compose probabilities from hierarchical heads.

        Supports legacy 2-head (3-class) and extended 3-head (K-class) variants.
        """
        if len(outputs) == 2:
            red_logits, yg_logits = outputs
            red_probs = torch.softmax(red_logits, dim=1)
            yg_probs = torch.softmax(yg_logits, dim=1)
            noncrit_gate = red_probs[:, 0:1]
            crit_gate = red_probs[:, 1:2]
            nc0 = noncrit_gate * yg_probs[:, 0:1]
            nc1 = noncrit_gate * yg_probs[:, 1:2]
            probs = torch.cat([nc0, nc1, crit_gate], dim=1)
            return probs.clamp_min(1e-8)
        elif len(outputs) == 3:
            gate_logits, nc_logits, crit_logits = outputs
            gate = torch.softmax(gate_logits, dim=1)
            noncrit = gate[:, 0:1]
            crit = gate[:, 1:2]
            nc_probs = torch.softmax(nc_logits, dim=1)
            crit_probs = torch.softmax(crit_logits, dim=1)
            left = noncrit * nc_probs
            right = crit * crit_probs
            probs = torch.cat([left, right], dim=1)
            return probs.clamp_min(1e-8)
        else:
            raise ValueError("Unsupported hierarchical outputs tuple length")

    # ---------------------------------------------------------
    def compute_hierarchical_loss_general(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        probs = self.compose_hierarchical_probs(outputs)
        # Now compute loss using composed probs
        log_probs = torch.log(probs)
        ce_loss = torch.nn.functional.nll_loss(log_probs, targets, weight=self.class_weights, reduction="mean")
        ce_raw = torch.nn.functional.nll_loss(log_probs, targets, reduction="none")
        pt = torch.exp(-ce_raw)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_raw
        focal_loss = focal_loss.mean()

        pred_classes = probs.argmax(dim=1)
        k = probs.size(-1)
        penalty_matrix = self._build_penalty_matrix(k, probs.device, probs.dtype)
        penalty_values = penalty_matrix[targets, pred_classes]
        safety_penalty = penalty_values.mean()

        if k == 3:
            critical_threshold = 2
        else:
            critical_threshold = max(1, k - 2)
        critical_mask = targets >= critical_threshold
        if critical_mask.any():
            critical_misses = (pred_classes[critical_mask] < critical_threshold).float().mean()
            critical_penalty = critical_misses * self.critical_miss_penalty
        else:
            critical_penalty = torch.tensor(0.0, device=probs.device)

        total_loss = (
            ce_loss
            + self.w_focal * focal_loss
            + self.w_safety * safety_penalty
            + self.w_critical * critical_penalty
        )
        return total_loss
