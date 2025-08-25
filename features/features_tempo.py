import pandas as pd
import numpy as np

def features_tempo_flex(df_tx: pd.DataFrame,
                        df_inad: pd.DataFrame,
                        id_col="id_cliente",
                        dt_col="data_transacao",
                        ref_col="data_referencia",
                        usar_M_1: bool = False) -> pd.DataFrame:
    """
    Gera variáveis de TEMPO relacionadas às transações.

    Inclui:
    - tempo_desde_primeira_Xm: dias desde a primeira transação no período até a data de referência
    - tempo_desde_ultima_Xm: dias desde a última transação no período até a data de referência
    (janelas: 1m, 3m, 6m, 12m, 24m, ever)

    Flags auxiliares:
    - flag_nunca_transacionou: cliente não possui nenhuma transação (ever vazio)
    - flag_sem_movimento_{X}m: cliente já teve transações, mas não naquela janela

    Convenções:
    - np.nan → não houve transação na janela
    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "12m": 12, "24m": 24, "ever": None}
    clientes_com_tx = set(df_tx[id_col].unique())

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                **{f"tempo_desde_primeira_{k}": np.nan for k in janelas.keys()},
                **{f"tempo_desde_ultima_{k}": np.nan for k in janelas.keys()},
                "flag_nunca_transacionou": 1,
                **{f"flag_sem_movimento_{k}": np.nan for k in janelas if k != "ever"}
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col] == cid) & (df_tx[dt_col] <= cutoff)]
        feats = {}

        # flag de nunca transacionou (ever)
        feats["flag_nunca_transacionou"] = 1 if tx_cliente.empty else 0

        for label, meses in janelas.items():
            if meses is None:  # ever
                tx_window = tx_cliente
            else:
                start = (cutoff - pd.DateOffset(months=meses-1)).replace(day=1)
                tx_window = tx_cliente[tx_cliente[dt_col] >= start]

            if tx_window.empty:
                feats[f"tempo_desde_primeira_{label}"] = np.nan
                feats[f"tempo_desde_ultima_{label}"] = np.nan
                if label != "ever":  # só faz sentido para janelas
                    feats[f"flag_sem_movimento_{label}"] = 1 if not tx_cliente.empty else 0
            else:
                primeira = tx_window[dt_col].min()
                ultima = tx_window[dt_col].max()
                feats[f"tempo_desde_primeira_{label}"] = (cutoff - primeira).days
                feats[f"tempo_desde_ultima_{label}"] = (cutoff - ultima).days
                if label != "ever":
                    feats[f"flag_sem_movimento_{label}"] = 0

        resultados.append({id_col: cid, ref_col: ref_date, **feats})

    return pd.DataFrame(resultados)
