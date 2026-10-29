# Copyright (c) 2026 takotime808

"""Multi-task GP surrogate (ICM) — optional dependency ``pip install gpytorch``.

Every other GP in the package fits one independent model per output. This one
fits all outputs jointly with the intrinsic coregionalization model (ICM)::

    Cov(f_i(x), f_j(x')) = k(x, x') * B[i, j],    B = W W^T + diag(v)

where ``W`` is ``n_outputs x rank``. ``B`` is learned by maximum likelihood
together with the input kernel and a separate noise level per output, so data
for one output inform the others in proportion to their learned correlation.

The model is written in the "Hadamard" form: training data are the observed
``(row, output)`` pairs, so ``Y`` may contain NaN where an output was not
observed (e.g. a simulation that failed partway) and the row still contributes
its other outputs. Exact inference costs O((n * m)^3) for ``n`` rows and ``m``
outputs, so it suits small designs (n * m up to a few thousand).
"""

from __future__ import annotations

import numpy as np

from multioutreg.surrogates.conformal_mixin import ConformalMixin

try:
    import torch
    import gpytorch
    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False


def _require_gpytorch() -> None:
    if not _GPYTORCH_AVAILABLE:
        raise ImportError(
            "gpytorch is required for MultiTaskGPSurrogate. "
            "Install it with: pip install gpytorch"
        )


_KERNELS = ("rbf", "matern52", "matern32")


class _HadamardICM(  # type: ignore[misc]
    gpytorch.models.ExactGP if _GPYTORCH_AVAILABLE else object  # type: ignore[misc]
):
    """Exact GP over (x, output index) pairs with an ICM kernel."""

    def __init__(self, train_i, train_x, train_y, likelihood, n_tasks, rank, kernel):
        # The output index goes first: at prediction time gpytorch passes the
        # model's train_inputs tuple to HadamardGaussianLikelihood, which reads
        # the task indices from element 0.
        super().__init__((train_i, train_x), train_y, likelihood)
        d = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        else:
            nu = 2.5 if kernel == "matern52" else 1.5
            self.covar_module = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=d)
        # IndexKernel holds B = W W^T + diag(v); it also sets the output scale,
        # so the input kernel has no ScaleKernel of its own.
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=rank)

    def forward(self, i, x):
        mean_x = self.mean_module(x)
        covar = self.covar_module(x).mul(self.task_covar_module(i))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class MultiTaskGPSurrogate(ConformalMixin):
    """Joint multi-output GP with a learned output correlation (ICM).

    Requires the optional ``gpytorch`` package::

        pip install gpytorch

    Parameters
    ----------
    rank : int, default 1
        Rank of the low-rank part of the output covariance ``B``. ``rank=1``
        lets every output share one latent function up to scale; higher ranks
        allow more independent structure. Clamped to the number of outputs.
    kernel : {"rbf", "matern52", "matern32"}, default "rbf"
        Input kernel, with one length scale per input (ARD).
    max_iter : int, default 200
        Adam steps for hyperparameter training.
    learning_rate : float, default 0.1
    predict_chunk : int, default 256
        Test rows predicted per batch. ``return_cov`` builds the joint
        covariance of ``predict_chunk * n_outputs`` points at a time.
    random_state : int | None, default None

    Attributes
    ----------
    output_corr_ : np.ndarray, shape (n_outputs, n_outputs)
        Learned correlation between the latent outputs (from ``B``).
    noise_std_ : np.ndarray, shape (n_outputs,)
        Learned observation-noise std per output, in output units.
    """

    def __init__(
        self,
        rank: int = 1,
        kernel: str = "rbf",
        max_iter: int = 200,
        learning_rate: float = 0.1,
        predict_chunk: int = 256,
        random_state: int | None = None,
    ):
        _require_gpytorch()
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")
        self.rank = rank
        self.kernel = kernel
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.predict_chunk = predict_chunk
        self.random_state = random_state

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiTaskGPSurrogate":
        """Fit the joint GP. ``Y`` may contain NaN for unobserved outputs."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, m = Y.shape
        self.n_outputs_ = m

        observed = ~np.isnan(Y)
        empty = np.flatnonzero(observed.sum(axis=0) == 0)
        if empty.size:
            raise ValueError(f"outputs {empty.tolist()} have no observed values")

        x_std = X.std(axis=0)
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(x_std == 0, 1.0, x_std)
        y_std = np.nanstd(Y, axis=0)
        self._y_mean = np.nanmean(Y, axis=0)
        self._y_std = np.where(y_std == 0, 1.0, y_std)

        rows, tasks = np.nonzero(observed)
        X_s = (X - self._x_mean) / self._x_std
        Y_s = (Y - self._y_mean) / self._y_std
        train_x = torch.as_tensor(X_s[rows], dtype=torch.float64)
        train_i = torch.as_tensor(tasks, dtype=torch.long).unsqueeze(-1)
        train_y = torch.as_tensor(Y_s[rows, tasks], dtype=torch.float64)

        likelihood = gpytorch.likelihoods.HadamardGaussianLikelihood(num_tasks=m)
        model = _HadamardICM(
            train_i, train_x, train_y, likelihood, m, max(1, min(self.rank, m)), self.kernel
        ).double()
        likelihood.double()

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            output = model(train_i, train_x)
            loss = -mll(output, train_y, train_i)
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()

        self._model = model
        self._likelihood = likelihood
        with torch.no_grad():
            B = model.task_covar_module.covar_matrix.to_dense().numpy()
            noise = likelihood.noise.reshape(-1).numpy()
        d = np.sqrt(np.diag(B))
        self.output_corr_ = B / np.outer(d, d)
        self._noise_var_s = noise  # in standardized units
        self.noise_std_ = np.sqrt(noise) * self._y_std
        return self

    def _predict_chunk(self, X_s, return_cov):
        n, m = len(X_s), self.n_outputs_
        x = torch.as_tensor(np.repeat(X_s, m, axis=0), dtype=torch.float64)
        i = torch.as_tensor(np.tile(np.arange(m), n), dtype=torch.long).unsqueeze(-1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self._model(i, x)
            mean = post.mean.numpy().reshape(n, m)
            if return_cov:
                full = post.covariance_matrix.numpy().reshape(n, m, n, m)
                idx = np.arange(n)
                cov = full[idx, :, idx, :]  # (n, m, m) within-row blocks
                cov[:, np.arange(m), np.arange(m)] += self._noise_var_s
                return mean, cov
            var = post.variance.clamp_min(0.0).numpy().reshape(n, m)
        return mean, var + self._noise_var_s

    def predict(self, X: np.ndarray, return_std: bool = False, return_cov: bool = False):
        """Predict all outputs.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        return_std : bool
            Also return the per-output predictive std (including noise).
        return_cov : bool
            Also return the predictive covariance between outputs for each
            row, shape ``(n_samples, n_outputs, n_outputs)``. Takes precedence
            over ``return_std``.
        """
        if not hasattr(self, "_model"):
            raise AttributeError("MultiTaskGPSurrogate is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        X_s = (X - self._x_mean) / self._x_std
        need_unc = return_std or return_cov

        means, uncs = [], []
        for start in range(0, len(X_s), max(1, self.predict_chunk)):
            mean, unc = self._predict_chunk(X_s[start:start + self.predict_chunk], return_cov)
            means.append(mean)
            uncs.append(unc)
        mean = np.concatenate(means) * self._y_std + self._y_mean
        if not need_unc:
            return mean
        if return_cov:
            cov = np.concatenate(uncs) * np.outer(self._y_std, self._y_std)
            return mean, cov
        std = np.sqrt(np.concatenate(uncs)) * self._y_std
        return mean, std

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(X)).reshape(len(X), -1)
