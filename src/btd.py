"""Bradley–Terry–Davidson para empates, implementado con NumPy/SciPy.

Para un par (i, j) con puntajes θ (en escala log) y parámetro de empate ν > 0:

    P(i gana) = e^{d/2} / Z,   P(j gana) = e^{-d/2} / Z,   P(empate) = ν / Z
    d = θ_i − θ_j,   Z = 2·cosh(d/2) + ν

Se estima por máxima verosimilitud penalizada (L2 suave sobre θ, para fijar la
escala y estabilizar modelos con pocas batallas) con L-BFGS y gradiente analítico.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

LABEL_A, LABEL_B, LABEL_TIE = 0, 1, 2


class BTD:
    def __init__(self, l2: float = 1e-2):
        self.l2 = l2
        self.theta_: dict[str, float] = {}
        self.nu_: float = 1.0

    # ── ajuste ────────────────────────────────────────────────────────────
    def fit(self, model_a, model_b, y) -> "BTD":
        a_names, b_names = np.asarray(model_a, dtype=object), np.asarray(model_b, dtype=object)
        y = np.asarray(y)
        names = sorted(set(a_names) | set(b_names))
        index = {n: i for i, n in enumerate(names)}
        ia = np.fromiter((index[n] for n in a_names), dtype=int, count=len(a_names))
        ib = np.fromiter((index[n] for n in b_names), dtype=int, count=len(b_names))
        k = len(names)
        is_a, is_b, is_t = (y == LABEL_A), (y == LABEL_B), (y == LABEL_TIE)
        sign = is_a.astype(float) - is_b.astype(float)   # +1 gana A, −1 gana B, 0 empate

        def objective(params):
            theta, u = params[:k], params[k]
            nu = np.exp(u)
            d = theta[ia] - theta[ib]
            z = 2.0 * np.cosh(d / 2.0) + nu
            ll = np.sum(sign * d / 2.0 + is_t * u - np.log(z))
            dlogz_dd = np.sinh(d / 2.0) / z
            g_d = sign / 2.0 - dlogz_dd
            g_theta = np.bincount(ia, g_d, k) - np.bincount(ib, g_d, k)
            g_u = np.sum(is_t - nu / z)
            loss = -ll + 0.5 * self.l2 * np.sum(theta ** 2)
            grad = np.concatenate([-g_theta + self.l2 * theta, [-g_u]])
            return loss, grad

        res = minimize(objective, np.zeros(k + 1), jac=True, method="L-BFGS-B")
        theta = res.x[:k] - res.x[:k].mean()
        self.theta_ = {n: float(theta[i]) for n, i in index.items()}
        self.nu_ = float(np.exp(res.x[k]))
        return self

    # ── predicción ────────────────────────────────────────────────────────
    def score(self, models) -> np.ndarray:
        """Puntaje θ; modelos no vistos reciben 0 (la media)."""
        return np.array([self.theta_.get(m, 0.0) for m in models], dtype=float)

    def predict_proba(self, model_a, model_b) -> np.ndarray:
        """Matriz (n, 3) con [P(A), P(B), P(empate)]."""
        d = self.score(model_a) - self.score(model_b)
        z = 2.0 * np.cosh(d / 2.0) + self.nu_
        return np.column_stack([np.exp(d / 2.0) / z, np.exp(-d / 2.0) / z, self.nu_ / z])

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.theta_.items(), key=lambda kv: -kv[1])
