import pandas as pd
import numpy as np

def features_quantidade_flex(df_tx: pd.DataFrame,
                             df_inad: pd.DataFrame,
                             id_col="id_cliente",
                             dt_col="data_transacao",
                             ref_col="data_referencia",
                             usar_M_1=False) -> pd.DataFrame:
    """
    Gera variáveis de QUANTIDADE de transações, com referência na base de inadimplência.

    Variáveis criadas:
    -----------------
    - qtde_trans_Xm : quantidade de transações no período (X = 1,3,6,9,12,24,ever)
    - pct_qtde_trans_Xm : proporção das transações no período X em relação ao total ever
    - comp_qtde_A_vs_B : razão entre qtde de transações de períodos vizinhos
    - delta_qtde_A_vs_B : diferença absoluta entre qtde de transações de períodos vizinhos

    Convenções:
    ------------------
    - NaN → cliente não tem nenhuma transação no histórico
    - np.nan → usado em divisões por zero
    - 0 → não houve transações no período analisado
    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "9m": 9, "12m": 12, "24m": 24, "ever": None}
    comparacoes = [("1m","3m"), ("3m","6m"), ("6m","9m"),("9m","12m"), ("12m","24m"), ("24m","ever")]

    clientes_com_tx = set(df_tx[id_col].unique())

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                **{f"qtde_trans_{k}": np.nan for k in janelas.keys()},
                **{f"pct_qtde_trans_{k}": np.nan for k in ["1m","3m","6m","12m","24m"]},
                **{f"comp_qtde_{a}_vs_{b}": np.nan for a,b in comparacoes},
                **{f"delta_qtde_{a}_vs_{b}": np.nan for a,b in comparacoes}
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col]==cid) & (df_tx[dt_col]<=cutoff)]
        feats = {}

        # Quantidade por janela
        for label, meses in janelas.items():
            if meses is None:
                tx_window = tx_cliente
            else:
                start = (cutoff - pd.DateOffset(months=meses-1)).replace(day=1)
                tx_window = tx_cliente[tx_cliente[dt_col] >= start]
            feats[f"qtde_trans_{label}"] = len(tx_window)

        # Proporções em relação ao total
        qtde_ever = feats["qtde_trans_ever"]
        for label in ["1m","3m","6m","12m","24m"]:
            v1 = feats[f"qtde_trans_{label}"]
            feats[f"pct_qtde_trans_{label}"] = (
                np.nan if qtde_ever == 0 else round(100*v1/qtde_ever,2)
            )

        # Comparações vizinhas
        for a,b in comparacoes:
            v1 = feats[f"qtde_trans_{a}"]
            v2 = feats[f"qtde_trans_{b}"]
            feats[f"comp_qtde_{a}_vs_{b}"] = np.nan if v2==0 else round(v1/v2,2)
            feats[f"delta_qtde_{a}_vs_{b}"] = v1-v2

        resultados.append({id_col: cid, ref_col: ref_date, **feats})

    return pd.DataFrame(resultados)
