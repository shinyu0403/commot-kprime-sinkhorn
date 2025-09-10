
import numpy as np
from scipy.spatial import distance_matrix

def ensure_unique_varnames(adata):
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

def get_distance_matrix(adata):
    if 'spatial_distance' in adata.obsp:
        return adata.obsp['spatial_distance']
    assert 'spatial' in adata.obsm, "adata.obsm['spatial'] is required to compute Euclidean distance"
    return distance_matrix(adata.obsm['spatial'], adata.obsm['spatial'])

def make_cost_matrix(D, thr, phi='euc'):
    if phi == 'euc_square':
        base = D**2
    else:
        base = D.copy()
    C = base.astype(float)
    if thr is not None:
        C[D > thr] = np.inf
    return C

def heteromeric_min(expr_matrix, name, delimiter='_'):
    parts = name.split(delimiter)
    if not set(parts).issubset(set(expr_matrix.var_names)):
        return None
    X = expr_matrix[:, parts].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X.min(axis=1).A1 if hasattr(X, 'A1') else X.min(axis=1)

def mono_gene(expr_matrix, name):
    if name not in expr_matrix.var_names:
        return None
    x = expr_matrix[:, name].X
    if hasattr(x, 'toarray'):
        x = x.toarray()
    return x.A1 if hasattr(x, 'A1') else x.ravel()
