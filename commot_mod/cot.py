
import numpy as np
from scipy import sparse
from .utils import make_cost_matrix, heteromeric_min, mono_gene
from .sinkhorn import sinkhorn_modified_Kprime
from .modulators import make_M, resolve_mod_vector

class CellCommunication:
    def __init__(self, adata, df_ligrec, dis_mat, dis_thr=None, cost_scale=None, cost_type='euc',
                 heteromeric=False, heteromeric_rule='min', heteromeric_delimiter='_'):
        self.adata = adata
        self.df_ligrec = df_ligrec
        self.dis_mat = dis_mat
        self.dis_thr = dis_thr
        self.cost_scale = cost_scale or {}
        self.cost_type = cost_type
        self.heteromeric = heteromeric
        self.heteromeric_rule = heteromeric_rule
        self.heteromeric_delimiter = heteromeric_delimiter
        self.ns = adata.n_obs

        self.ligs = list(np.unique(df_ligrec.iloc[:,0].values))
        self.recs = list(np.unique(df_ligrec.iloc[:,1].values))
        self.lig_index = {g:i for i,g in enumerate(self.ligs)}
        self.rec_index = {g:j for j,g in enumerate(self.recs)}
        self.comm_network = {}

    def _get_vec(self, gene, is_ligand):
        if self.heteromeric and ('_' in gene):
            v = heteromeric_min(self.adata, gene, delimiter=self.heteromeric_delimiter)
        else:
            v = mono_gene(self.adata, gene)
        if v is None:
            return None
        v = np.maximum(np.asarray(v, dtype=float).ravel(), 0.0)
        return v

    def _pair_cost(self, i_name, j_name):
        C = make_cost_matrix(self.dis_mat, self.dis_thr, phi='euc' if self.cost_type == 'euc' else 'euc_square')
        if self.cost_scale and (i_name, j_name) in self.cost_scale:
            C = C * float(self.cost_scale[(i_name, j_name)])
        return C

    def run_cot_signaling(self, cot_eps_p=1e-1, cot_eps_mu=None, cot_eps_nu=None, cot_rho=1e1,
                          cot_nitermax=10000, cot_weights=(0.25,0.25,0.25,0.25),
                          smooth=False, smth_eta=None, smth_nu=None, smth_kernel='exp',
                          modulators=None, Kh=0.5):
        ns = self.ns
        for _, (lig_name, rec_name, _) in self.df_ligrec.iterrows():
            xL = self._get_vec(lig_name, is_ligand=True)
            xR = self._get_vec(rec_name, is_ligand=False)
            if xL is None or xR is None:
                continue
            a = xL.copy()
            b = xR.copy()
            C = self._pair_cost(lig_name, rec_name)

            AG = modulators.get('AG', {}).get(lig_name) if modulators is not None else None
            AN = modulators.get('AN', {}).get(lig_name) if modulators is not None else None
            CS = modulators.get('CS', {}).get(rec_name) if modulators is not None else None
            CI = modulators.get('CI', {}).get(rec_name) if modulators is not None else None
            M = make_M(a, b, AG=AG, AN=AN, CS=CS, CI=CI, Kh=Kh) if modulators is not None else None

            P = sinkhorn_modified_Kprime(C, a, b, eps=cot_eps_p, rho=cot_rho, M=M, n_iter=cot_nitermax)

            S = sparse.csr_matrix(P)
            i = self.lig_index.setdefault(lig_name, len(self.lig_index))
            j = self.rec_index.setdefault(rec_name, len(self.rec_index))
            self.comm_network[(i,j)] = S
    
    def _pair_cost_matrix(self, i_idx, j_idx):
        """
        由 self.M（距離或平方距離）複製出一份 C_{ij}，超過該 pair 的 cutoff 一律設為 +inf，
        再乘上 A 的權重（若 A[i,j] 是 inf 代表這對無效）。
        """
        C = self.M.copy().astype(float)
        thr = self.cutoff[i_idx, j_idx] if hasattr(self, "cutoff") else None
        if thr is not None and np.isfinite(thr):
            C[C > thr] = np.inf
        w = self.A[i_idx, j_idx]
        if np.isfinite(w):
            C = C * float(w)
        else:
            # 無效 pair：全部設 inf
            C[:] = np.inf
        return C

    def _pair_modulation_M(self, i_idx, j_idx, modulators=None, Kh=0.5):
        if modulators is None:
            return None
        ns = self.npts
        lig_name = self.ligs[i_idx]
        rec_name = self.recs[j_idx]

        AG = resolve_mod_vector(lig_name, modulators.get("AG"), npts=ns,
                                heteromeric=self.heteromeric, rule=self.heteromeric_rule,
                                delimiter=self.heteromeric_delimiter)
        AN = resolve_mod_vector(lig_name, modulators.get("AN"), npts=ns,
                                heteromeric=self.heteromeric, rule=self.heteromeric_rule,
                                delimiter=self.heteromeric_delimiter)
        CS = resolve_mod_vector(rec_name, modulators.get("CS"), npts=ns,
                                heteromeric=self.heteromeric, rule=self.heteromeric_rule,
                                delimiter=self.heteromeric_delimiter)
        CI = resolve_mod_vector(rec_name, modulators.get("CI"), npts=ns,
                                heteromeric=self.heteromeric, rule=self.heteromeric_rule,
                                delimiter=self.heteromeric_delimiter)

        # 注意：make_M 只拿 lig_vec/rec_vec 來決定 ns，傳入 S/D 的一欄即可
        lig_vec = self.S[:, i_idx]
        rec_vec = self.D[:, j_idx]
        return make_M(lig_vec, rec_vec, AG=AG, AN=AN, CS=CS, CI=CI, Kh=Kh)

    def run_cot_signaling_modified(self,
        cot_eps_p=1e-1, cot_rho=1e1, cot_nitermax=10000,
        modulators=None, Kh=0.5
    ):
        """
        用『修改後 Sinkhorn』與 M 調節去計算每對 LR 的 cell×cell 傳輸矩陣，
        結果存進 self.comm_network[(i,j)]（csr_matrix），外層即可沿用原寫回 adata 的程式。
        """
        from scipy import sparse
        for i_idx in range(self.nlig):
            a = self.S[:, i_idx].astype(float).ravel()  # sender marginal
            for j_idx in range(self.nrec):
                b = self.D[:, j_idx].astype(float).ravel()  # receiver marginal
                # 無效 pair 直接跳過
                if not np.isfinite(self.A[i_idx, j_idx]):
                    continue
                Cij = self._pair_cost_matrix(i_idx, j_idx)
                Mij = self._pair_modulation_M(i_idx, j_idx, modulators=modulators, Kh=Kh) if modulators is not None else None
                P = sinkhorn_modified_Kprime(Cij, a, b, eps=cot_eps_p, rho=cot_rho, M=Mij, n_iter=cot_nitermax)
                self.comm_network[(i_idx, j_idx)] = sparse.csr_matrix(P)
