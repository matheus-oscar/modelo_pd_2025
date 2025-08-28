import pandas as pd
import numpy as np


def features_flags_flex(df_tx: pd.DataFrame,
                        df_inad: pd.DataFrame,
                        id_col="id_cliente",
                        dt_col="data_transacao",
                        ref_col="data_referencia",
                        usar_M_1: bool = False) -> pd.DataFrame:
    """
    Gera variáveis de FLAGS de existência de transação.

    Variáveis criadas:
    - flag_nunca_transacionou: 1 se o cliente nunca teve transações no histórico, 0 caso contrário
    - flag_transacao_Xm: indica se o cliente teve pelo menos uma transação na janela
        * 1 → cliente teve transação
        * 0 → cliente tem histórico, mas não nessa janela
        * NaN → cliente nunca transacionou

    Janelas: 1m, 3m, 6m, 9m, 12m, 24m, ever
    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "9m": 9,
               "12m": 12, "24m": 24, "ever": None}
    clientes_com_tx = set(df_tx[id_col].unique())

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                "flag_nunca_transacionou": 1,
                **{f"flag_transacao_{k}": np.nan for k in janelas.keys()}
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col] == cid) & (df_tx[dt_col] <= cutoff)]
        feats = {}
        feats["flag_nunca_transacionou"] = 1 if tx_cliente.empty else 0

        for label, meses in janelas.items():
            if meses is None:
                tx_window = tx_cliente
            else:
                start = (cutoff - pd.DateOffset(months=meses - 1)).replace(day=1)
                tx_window = tx_cliente[tx_cliente[dt_col] >= start]

            if tx_window.empty:
                feats[f"flag_transacao_{label}"] = np.nan if tx_cliente.empty else 0
            else:
                feats[f"flag_transacao_{label}"] = 1

        resultados.append({id_col: cid, ref_col: ref_date, **feats})

    return pd.DataFrame(resultados)
