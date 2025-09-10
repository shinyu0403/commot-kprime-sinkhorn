

import numpy as np

def _row_lse(logKp, g, inv_eps):
    X = logKp + g[None, :] * inv_eps          # (ns, nt)
    m = np.max(X, axis=1, keepdims=True)
    return (np.log(np.sum(np.exp(X - m), axis=1)) + m.ravel())  # shape (ns,)

def _col_lse(logKp, f, inv_eps):
    X = logKp.T + f[None, :] * inv_eps        # (nt, ns)
    m = np.max(X, axis=1, keepdims=True)
    return (np.log(np.sum(np.exp(X - m), axis=1)) + m.ravel())  # shape (nt,)

def _lse_rows_safe(X):
    # X: (n, m) 返還: 長度 n
    m = np.max(X, axis=1)          # shape (n,)
    bad = ~np.isfinite(m)          # 該行全為 -inf
    m = m[:, None]
    Y = np.exp(X - m)
    Y[~np.isfinite(Y)] = 0.0
    s = np.log(np.sum(Y, axis=1))
    s[bad] = -np.inf
    return s + m.ravel()

def unot_sinkhorn_l1_dense_modified(a, b, C, eps, m, *, M=None,
                                    nitermax=10000, stopthr=1e-8, verbose=False):
    """
    L1 懲罰的穩定化 Sinkhorn（修改版），使用 K' = M * exp(-C/eps)。
    """
    ns, nt = C.shape
    if M is None:
        M = np.ones_like(C, float)

    # 合法位置：成本有限 + M>0
    mask = np.isfinite(C) & (M > 0)

    # logK' = logM - C/eps
    inv_eps = 1.0 / float(eps)
    Mpos = np.maximum(M, 0.0)
    logM = np.full_like(Mpos, -np.inf, float)
    logM[Mpos > 0] = np.log(Mpos[Mpos > 0])
    logKp = np.full_like(C, -np.inf, float)
    logKp[mask] = logM[mask] - C[mask] * inv_eps

    # 邊際避免 log(0)
    a = np.maximum(np.asarray(a, float).ravel(), 1e-300)
    b = np.maximum(np.asarray(b, float).ravel(), 1e-300)

    f = np.zeros(ns, float)
    g = np.zeros(nt, float)

    for it in range(int(nitermax)):
        f_prev, g_prev = f.copy(), g.copy()

        # row term: log sum_j exp((f_i + g_j - C_ij)/eps)
        row_term = (f * inv_eps) + _lse_rows_safe(logKp + g[None, :] * inv_eps)
        l_unmatch_f = (f - m) * inv_eps
        f = eps * np.log(a) + f - eps * np.logaddexp(row_term, l_unmatch_f)

        # col term: log sum_i exp((f_i + g_j - C_ij)/eps)
        col_term = (g * inv_eps) + _lse_rows_safe((logKp.T + f[None, :] * inv_eps))
        l_unmatch_g = (g - m) * inv_eps
        g = eps * np.log(b) + g - eps * np.logaddexp(col_term, l_unmatch_g)

        if it % 10 == 0:
            err = max(np.max(np.abs(f - f_prev)), np.max(np.abs(g - g_prev)))
            if err < float(stopthr):
                break

    expo = (f[:, None] + g[None, :] - C) * inv_eps
    expo[~mask] = -np.inf
    P = M * np.exp(expo)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P[P < 0] = 0.0
    if verbose:
        print(f"[K' Sinkhorn] it={it}, P.sum={P.sum():.6g}")
    return P


def sinkhorn_modified_Kprime(C, a, b, *, eps=1e-1, rho=1e1, M=None, n_iter=2000, tol=1e-7): 
    """
    V1
    Modified unbalanced Sinkhorn with your K' = M * exp(-C/eps).
    C  : (ns, ns) cost (+inf outside threshold)
    a  : (ns,) sender marginal (ligand expression per cell)
    b  : (ns,) receiver marginal (receptor expression per cell)
    M  : (ns, ns) >=0 modulation matrix; if None, defaults to 1
    """
    ns = C.shape[0]
    if M is None:
        M = np.ones_like(C, dtype=float)

    # valid positions: finite C and positive M
    mask = np.isfinite(C) & (M > 0)

    # log K' = log M - C/eps  (else -inf)
    logKp = np.full_like(C, -np.inf, dtype=float)
    logKp[mask] = np.log(M[mask]) - C[mask] / float(eps)

    f = np.zeros(ns, dtype=float)
    g = np.zeros(ns, dtype=float)
    inv_eps = 1.0 / float(eps)

    def row_lse(logKp, g):
        X = logKp + g[None, :] * inv_eps
        m = np.max(X, axis=1, keepdims=True)
        return (np.log(np.sum(np.exp(X - m), axis=1)) + m.ravel())

    def col_lse(logKp, f):
        X = logKp.T + f[None, :] * inv_eps
        m = np.max(X, axis=1, keepdims=True)
        return (np.log(np.sum(np.exp(X - m), axis=1)) + m.ravel())

    for _ in range(int(n_iter)):
        s1 = row_lse(logKp, g)                          # log(K' e^{g/eps})
        l_unmatched_f = (f - rho) * inv_eps             # log e^{(f-ρ)/ε}
        m = np.maximum(s1, l_unmatched_f)
        f_new = eps * np.log(np.maximum(a, 1e-300)) + f \
                - eps * (np.log(np.exp(s1 - m) + np.exp(l_unmatched_f - m)) + m)

        s2 = col_lse(logKp, f_new)                      # log((K')^T e^{f/eps})
        l_unmatched_g = (g - rho) * inv_eps
        m2 = np.maximum(s2, l_unmatched_g)
        g_new = eps * np.log(np.maximum(b, 1e-300)) + g \
                - eps * (np.log(np.exp(s2 - m2) + np.exp(l_unmatched_g - m2)) + m2)

        if max(np.max(np.abs(f_new-f)), np.max(np.abs(g_new-g))) < tol:
            f, g = f_new, g_new
            break
        f, g = f_new, g_new

    # P* = M ⊙ exp((f ⊕ g - C)/ε)  (禁用位置設 0)
    expo = (f[:, None] + g[None, :] - C) * inv_eps
    expo[~mask] = -np.inf
    P = M * np.exp(expo)
    return P

def uot_sinkhorn_scaling(C, a, b, *, eps=1.0, rho=1.0, M=None, n_iter=500, tol=1e-9):
    """V2"""
    ns = C.shape[0]
    if M is None:
        M = np.ones_like(C, dtype=float)
    K = np.exp(-C / float(eps)) * M
    K[~np.isfinite(C)] = 0.0
    K[M <= 0] = 0.0
    a = np.maximum(a.astype(float).ravel(), 1e-300)
    b = np.maximum(b.astype(float).ravel(), 1e-300)
    u = np.ones(ns, float); v = np.ones(ns, float)
    tau = float(eps) / (float(eps) + float(rho))
    for _ in range(int(n_iter)):
        Kv = K @ v + 1e-300
        u_new = (a / Kv) ** tau
        KTu = K.T @ u_new + 1e-300
        v_new = (b / KTu) ** tau
        if max(np.max(np.abs(u_new-u)), np.max(np.abs(v_new-v))) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    P = (u[:, None] * K) * v[None, :]
    return P