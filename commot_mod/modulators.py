
import numpy as np

def hill_ratio_pos(x, Kh):
    return 1.0 + np.maximum(x, 0.0) / float(Kh)

def hill_ratio_neg(x, Kh):
    return float(Kh) / (float(Kh) + np.maximum(x, 0.0) + 1e-12)

def make_M(lig_vec, rec_vec, *, AG=None, AN=None, CS=None, CI=None, Kh=0.5):
    ns = lig_vec.shape[0]
    s_ag = hill_ratio_pos(AG, Kh) if AG is not None else 1.0
    s_an = hill_ratio_neg(AN, Kh) if AN is not None else 1.0
    r_cs = hill_ratio_pos(CS, Kh) if CS is not None else 1.0
    r_ci = hill_ratio_neg(CI, Kh) if CI is not None else 1.0

    S = np.outer(np.atleast_1d(s_ag) * np.atleast_1d(s_an), np.ones(ns))
    R = np.outer(np.ones(ns), np.atleast_1d(r_cs) * np.atleast_1d(r_ci))
    M = S * R
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M[M < 0] = 0.0
    return M

def resolve_mod_vector(
    name: str,
    dct: dict | None,
    *,
    npts: int,
    heteromeric: bool,
    rule: str = "min",
    delimiter: str = "_",
):
    """
    從 dct（鍵=基因或複合體名, 值=|spots|向量）取出 'name' 的調節向量。
    1) 若有「複合體名」鍵，直接回傳；
    2) 否則若是異質複合體，對各組成基因的向量依 rule ('min'/'ave') 聚合；
    3) 都沒有則回傳 None。
    """
    if not dct:
        return None
    # 直接有複合體名
    if name in dct:
        v = np.asarray(dct[name], float).ravel()
        assert v.size == npts, f"{name} 向量長度 ({v.size}) 必須等於 npts={npts}"
        return np.maximum(v, 0.0)

    # 組成基因聚合
    if heteromeric and (delimiter in name):
        parts = name.split(delimiter)
        vecs = []
        for p in parts:
            vp = dct.get(p)
            if vp is None:
                return None  # 少任一個就不聚合，視為無此側調節
            vp = np.asarray(vp, float).ravel()
            assert vp.size == npts, f"{p} 向量長度 ({vp.size}) 必須等於 npts={npts}"
            vecs.append(np.maximum(vp, 0.0))
        V = np.vstack(vecs)
        return V.mean(0) if rule == "ave" else V.min(0)
    return None