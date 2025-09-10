
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy import sparse
import anndata
from .cot import CellCommunication
from .utils import get_distance_matrix

def spatial_communication(
    adata: anndata.AnnData, 
    database_name: str = None, 
    df_ligrec: pd.DataFrame = None,
    pathway_sum: bool = False,
    heteromeric: bool = False,
    heteromeric_rule: str = 'min',
    heteromeric_delimiter: str = '_', 
    dis_thr: Optional[float] = None, 
    cost_scale: Optional[dict] = None, 
    cost_type: str = 'euc',
    cot_eps_p: float = 1e-1, 
    cot_eps_mu: Optional[float] = None, 
    cot_eps_nu: Optional[float] = None, 
    cot_rho: float =1e1, 
    cot_nitermax: int = 10000, 
    cot_weights: Tuple[float,float,float,float] = (0.25,0.25,0.25,0.25), 
    smooth: bool = False, 
    smth_eta: float = None, 
    smth_nu: float = None, 
    smth_kernel: str = 'exp',
    modulators: Optional[dict] = None,
    hill_Kh: float = 0.5,
    copy: bool = False  
):
    assert database_name is not None, "Please give a database_name"
    assert df_ligrec is not None, "Please give a ligand-receptor database"

    dis_mat = get_distance_matrix(adata)

    data_genes = set(list(adata.var_names))
    tmp = []
    for i in range(df_ligrec.shape[0]):
        lig = str(df_ligrec.iloc[i][0])
        rec = str(df_ligrec.iloc[i][1])
        if heteromeric:
            lig_ok = set(lig.split(heteromeric_delimiter)).issubset(data_genes)
            rec_ok = set(rec.split(heteromeric_delimiter)).issubset(data_genes)
        else:
            lig_ok = lig in data_genes
            rec_ok = rec in data_genes
        if lig_ok and rec_ok:
            row = [lig, rec]
            if df_ligrec.shape[1] >= 3:
                row.append(df_ligrec.iloc[i][2])
            else:
                row.append('NA')
            tmp.append(row)
    df_ligrec = pd.DataFrame(tmp, columns=['ligand','receptor','pathway']).drop_duplicates()

    model = CellCommunication(
        adata,
        df_ligrec[['ligand','receptor','pathway']],
        dis_mat,
        dis_thr,
        cost_scale,
        cost_type,
        heteromeric=heteromeric,
        heteromeric_rule=heteromeric_rule,
        heteromeric_delimiter=heteromeric_delimiter
    )

    model.run_cot_signaling(
        cot_eps_p=cot_eps_p,
        cot_eps_mu=cot_eps_mu,
        cot_eps_nu=cot_eps_nu,
        cot_rho=cot_rho,
        cot_nitermax=cot_nitermax,
        cot_weights=cot_weights,
        smooth=smooth,
        smth_eta=smth_eta,
        smth_nu=smth_nu,
        smth_kernel=smth_kernel,
        modulators=modulators,
        Kh=hill_Kh
    )

    adata.uns['commot-'+database_name+'-info'] = {}
    adata.uns['commot-'+database_name+'-info']['df_ligrec'] = df_ligrec
    adata.uns['commot-'+database_name+'-info']['distance_threshold'] = dis_thr

    ncell = adata.shape[0]
    X_sender = np.empty([ncell,0], float)
    X_receiver = np.empty([ncell,0], float)
    col_names_sender = []
    col_names_receiver = []
    tmp_ligs = model.ligs
    tmp_recs = model.recs
    S_total = sparse.csr_matrix((ncell, ncell), dtype=float)
    pathways = sorted(df_ligrec['pathway'].unique().tolist()) if pathway_sum else []
    S_pathway = [sparse.csr_matrix((ncell, ncell), dtype=float) for _ in pathways]
    X_sender_pathway = [np.zeros([ncell,1], float) for _ in pathways]
    X_receiver_pathway = [np.zeros([ncell,1], float) for _ in pathways]

    for (i,j), S in model.comm_network.items():
        lig_name = tmp_ligs[i]
        rec_name = tmp_recs[j]
        adata.obsp['commot-'+database_name+'-'+lig_name+'-'+rec_name] = S
        S_total = S_total + S
        lig_sum = np.array(S.sum(axis=1))
        rec_sum = np.array(S.sum(axis=0).T)
        X_sender = np.concatenate((X_sender, lig_sum), axis=1)
        X_receiver = np.concatenate((X_receiver, rec_sum), axis=1)
        col_names_sender.append(f"s-{lig_name}-{rec_name}")
        col_names_receiver.append(f"r-{lig_name}-{rec_name}")
        if pathway_sum:
            mask = (df_ligrec['ligand'] == lig_name) & (df_ligrec['receptor'] == rec_name)
            pathway = df_ligrec[mask]['pathway'].values[0]
            idx = pathways.index(pathway)
            S_pathway[idx] = S_pathway[idx] + S
            X_sender_pathway[idx] = X_sender_pathway[idx] + np.array(S.sum(axis=1))
            X_receiver_pathway[idx] = X_receiver_pathway[idx] + np.array(S.sum(axis=0).T)

    if pathway_sum:
        for idx, pathway in enumerate(pathways):
            adata.obsp['commot-'+database_name+'-'+pathway] = S_pathway[idx]

    X_sender = np.concatenate((X_sender, X_sender.sum(axis=1).reshape(-1,1)), axis=1)
    X_receiver = np.concatenate((X_receiver, X_receiver.sum(axis=1).reshape(-1,1)), axis=1)
    col_names_sender.append("s-total-total")
    col_names_receiver.append("r-total-total")

    if pathway_sum:
        for idx, pathway in enumerate(pathways):
            X_sender = np.concatenate((X_sender, X_sender_pathway[idx]), axis=1)
            X_receiver = np.concatenate((X_receiver, X_receiver_pathway[idx]), axis=1)
            col_names_sender.append("s-"+pathway)
            col_names_receiver.append("r-"+pathway)

    adata.obsp['commot-'+database_name+'-total-total'] = S_total
    adata.obsm['commot-'+database_name+'-sum-sender'] = pd.DataFrame(data=X_sender, columns=col_names_sender, index=adata.obs_names)
    adata.obsm['commot-'+database_name+'-sum-receiver'] = pd.DataFrame(data=X_receiver, columns=col_names_receiver, index=adata.obs_names)

    return adata if copy else None
