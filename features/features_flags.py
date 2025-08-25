import pandas as pd
import numpy as np

def features_flags_flex(df_tx: pd.DataFrame,
                        df_inad: pd.DataFrame,
                        id_col="id_cliente",
                        dt_col="data_transacao",
                        ref_col="data_referencia",
                        usar_M_1: bool = False) -> pd.DataFrame:
    """
    Gera variáveis de FLAGS de existência de transação em diferentes janelas.

    Regras de cutoff
    ----------------
    - usar_M_1 = False → considera transações até o fim do próprio mês da data de referência (M).
    - usar_M_1 = True  → considera transações até o fim do mês anterior à data de referência (M-1).

    Variáveis criadas
    -----------------
    - flag_transacao_Xm:
        * 1 → cliente teve pelo menos uma transação na janela
        * 0 → cliente já teve histórico, mas não nessa janela
        * NaN → não aplicável (cliente nunca transacionou)
    - flag_nunca_transacionou: 1 se o cliente não possui transações no histórico (ever), 0 caso contrário
    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "12m": 12, "24m": 24, "ever": None}
    clientes_com_tx = set(df_tx[id_col].unique())

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]

        # cutoff: M ou M-1
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                **{f"flag_transacao_{k}": np.nan for k in janelas.keys()},
                "flag_nunca_transacionou": 1
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col] == cid) & (df_tx[dt_col] <= cutoff)]
        feats = {}

        feats["flag_nunca_transacionou"] = 1 if tx_cliente.empty else 0

        # Flags por janela
        for label, meses in janelas.items():
            if meses is None:  # ever
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
