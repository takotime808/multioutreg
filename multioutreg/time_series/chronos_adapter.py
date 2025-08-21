# Copyright (c) 2025 takotime808

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    import torch
    from chronos import BaseChronosPipeline, ChronosPipeline
except Exception as e:  # pragma: no cover - makes import optional
    BaseChronosPipeline = ChronosPipeline = None
    _import_err = e
else:
    _import_err = None


@dataclass
class ForecastResult:
    """Container for forecast outputs."""
    # shape: [n_series, n_quantiles, horizon]
    quantiles: np.ndarray
    # the quantile levels actually returned by the model
    q_levels: Tuple[float, ...]
    # optional: per-series identifiers (e.g., target names)
    ids: Tuple[str, ...] = ()
    # mean/median convenience views
    @property
    def median(self) -> np.ndarray:
        if 0.5 in self.q_levels:
            i = self.q_levels.index(0.5)
            return self.quantiles[:, i, :]
        # fallback to mid-quantile average
        return self.quantiles.mean(axis=1)


class ChronosForecaster:
    """
    Thin, sklearn-ish wrapper around Chronos / Chronos-Bolt.

    - For *Bolt* models (e.g., 'amazon/chronos-bolt-base'): direct multi-step *quantile* forecasting.
    - For original *Chronos* (e.g., 'amazon/chronos-t5-small'): we sample trajectories and compute quantiles.

    References: Chronos paper & repo.  """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-base",
        device: str | None = None,                # "cuda", "mps", or "cpu"
        torch_dtype: Optional["torch.dtype"] = None,  # e.g., torch.bfloat16
        context_length: Optional[int] = None,     # None => let pipeline decide
        default_quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ):
        if BaseChronosPipeline is None:
            raise ImportError(
                "chronos-forecasting not available. Install with `pip install .[ts]`.\n"
                f"Original import error: {_import_err}"
            )
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.torch_dtype = torch_dtype or (getattr(torch, "bfloat16", torch.float32) if self.device != "cpu" else torch.float32)
        self.context_length = context_length
        self.default_quantiles = tuple(float(q) for q in default_quantiles)

        self._is_bolt = "chronos-bolt" in model_name
        self._pipe = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )

        # storage
        self._series_ids: Tuple[str, ...] = ()
        self._contexts: list[torch.Tensor] = []

    # ---- public API --------------------------------------------------------

    def fit(self, y: pd.Series | np.ndarray | pd.DataFrame | Dict[str, Iterable[float]]) -> "ChronosForecaster":
        """
        Fit simply records the historical context (Chronos is pretrained / zero-shot).
        Accepts:
          - 1D array/Series (single univariate series)
          - DataFrame with 1+ columns (multi-target => per-column series)
          - dict of {name: sequence}
        """
        series: Dict[str, np.ndarray]

        if isinstance(y, dict):
            series = {str(k): np.asarray(v, dtype=float).ravel() for k, v in y.items()}
        elif isinstance(y, pd.DataFrame):
            series = {str(c): y[c].to_numpy(dtype=float).ravel() for c in y.columns}
        else:
            arr = pd.Series(y).to_numpy(dtype=float).ravel()
            series = {"y": arr}

        self._series_ids = tuple(series.keys())
        self._contexts = [torch.tensor(v, dtype=torch.float32) for v in series.values()]
        return self

    @torch.no_grad()
    def predict(
        self,
        prediction_length: int,
        quantiles: Optional[Sequence[float]] = None,
        num_samples: int = 200,  # used only for original Chronos (not Bolt)
    ) -> ForecastResult:
        """
        Returns quantile forecasts with shape [n_series, n_quantiles, horizon].
        """
        if not self._contexts:
            raise RuntimeError("Call fit(...) with your historical series first.")

        q_levels = tuple(float(q) for q in (quantiles or self.default_quantiles))
        batch = self._left_pad(self._contexts)

        if self._is_bolt:
            # direct quantile output: [B, Q, H]
            out = self._pipe.predict(context=batch, prediction_length=int(prediction_length), quantiles=q_levels)
            q = out.detach().cpu().numpy()
            return ForecastResult(quantiles=q, q_levels=q_levels, ids=self._series_ids)

        # original Chronos: sample trajectories then take quantiles
        pipe = ChronosPipeline.from_pretrained(  # type: ignore[attr-defined]
            self.model_name, device_map=self.device, torch_dtype=self.torch_dtype
        )
        samples = pipe.predict(  # [B, S, H]
            context=batch, prediction_length=int(prediction_length), num_samples=int(num_samples)
        )
        samples = samples.detach().cpu().numpy()
        q = np.quantile(samples, q_levels, axis=1)  # [Q, B, H]
        q = np.transpose(q, (1, 0, 2))              # [B, Q, H]
        return ForecastResult(quantiles=q, q_levels=q_levels, ids=self._series_ids)

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _left_pad(contexts: Sequence["torch.Tensor"]) -> "torch.Tensor":
        """Left-pad 1D tensors to same length: shape [B, T]."""
        import torch
        T = max(int(x.shape[0]) for x in contexts)
        out = []
        for x in contexts:
            pad = T - int(x.shape[0])
            if pad > 0:
                out.append(torch.nn.functional.pad(x, (pad, 0)))
            else:
                out.append(x)
        return torch.stack(out, dim=0)
