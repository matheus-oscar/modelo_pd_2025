import pandas as pd
import numpy as np


def features_valor_flex(df_tx: pd.DataFrame,
                        df_inad: pd.DataFrame,
                        id_col="id_cliente",
                        val_col="valor_transacao",
                        dt_col="data_transacao",
                        ref_col="data_referencia",
                        usar_M_1=False) -> pd.DataFrame:
    """
    Gera variáveis de VALOR a partir da base de transações,
    com referência na base de inadimplência.

    Lógicas:
    - Considera todas as transações até a data de cutoff:
        * usar_M_1=False → cutoff = data_referencia (fim do próprio mês, M)
        * usar_M_1=True  → cutoff = último dia do mês anterior (M-1)

    Variáveis criadas:
    -----------------
    - vlr_trans_Xm : soma do valor transacionado no período (X = 1,3,6,9,12,24,ever)
    - vlr_trans_ult : valor da última transação até o cutoff
    - vlr_trans_max : maior valor de transação até o cutoff
    - vlr_trans_min : menor valor de transação até o cutoff
    - comp_vlr_A_vs_B : razão entre valores transacionados em períodos vizinhos
    - delta_vlr_A_vs_B : diferença absoluta de valores entre períodos vizinhos

    Convenções:
    ------------------
    - NaN → cliente não tem nenhuma transação no histórico
    - -1  → denominador da comparação (período mais longo) é zero e o período curto tem valor > 0
    - 0   → ambos os períodos não têm valor
    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "9m": 9,
               "12m": 12, "24m": 24, "ever": None}
    comparacoes = [("1m", "3m"), ("3m", "6m"), ("6m", "9m"),
                   ("9m", "12m"), ("12m", "24m"), ("24m", "ever")]

    clientes_com_tx = set(df_tx[id_col].unique())

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                **{f"vlr_trans_{k}": np.nan for k in janelas.keys()},
                "vlr_trans_ult": np.nan,
                "vlr_trans_max": np.nan,
                "vlr_trans_min": np.nan,
                **{f"comp_vlr_{a}_vs_{b}": np.nan for a, b in comparacoes},
                **{f"delta_vlr_{a}_vs_{b}": np.nan for a, b in comparacoes}
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col] == cid) & (df_tx[dt_col] <= cutoff)]
        feats = {}

        # Totais por janela
        for label, meses in janelas.items():
            if meses is None:
                tx_window = tx_cliente
            else:
                start = (cutoff - pd.DateOffset(months=meses-1)).replace(day=1)
                tx_window = tx_cliente[tx_cliente[dt_col] >= start]
            feats[f"vlr_trans_{label}"] = tx_window[val_col].sum(
            ) if not tx_window.empty else 0

        # Última, máxima e mínima com idxmax/idxmin
        if not tx_cliente.empty:
            idx_ult = tx_cliente[dt_col].idxmax()
            feats["vlr_trans_ult"] = tx_cliente.loc[idx_ult, val_col]

            idx_max = tx_cliente[val_col].idxmax()
            feats["vlr_trans_max"] = tx_cliente.loc[idx_max, val_col]

            idx_min = tx_cliente[val_col].idxmin()
            feats["vlr_trans_min"] = tx_cliente.loc[idx_min, val_col]
        else:
            feats["vlr_trans_ult"] = np.nan
            feats["vlr_trans_max"] = np.nan
            feats["vlr_trans_min"] = np.nan

        # Comparações vizinhas
        for a, b in comparacoes:
            v1 = feats[f"vlr_trans_{a}"]
            v2 = feats[f"vlr_trans_{b}"]
            feats[f"comp_vlr_{a}_vs_{b}"] = np.nan if (
                v1 == 0 and v2 == 0) else (-1 if v2 == 0 else round(v1/v2, 3))
            feats[f"delta_vlr_{a}_vs_{b}"] = v1 - v2

        resultados.append({id_col: cid, ref_col: ref_date, **feats})

    return pd.DataFrame(resultados)
