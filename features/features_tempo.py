import pandas as pd
import numpy as np


def features_tempo_flex(df_tx: pd.DataFrame,
                        df_inad: pd.DataFrame,
                        id_col: str = "id_cliente",
                        dt_col: str = "data_transacao",
                        ref_col: str = "data_referencia",
                        usar_M_1: bool = True) -> pd.DataFrame:
    """
    Gera variáveis de TEMPO relacionadas às transações de clientes,
    alinhadas em janelas mensais fechadas.

    -------------------------
    Lógica de cutoff:
    -------------------------
    - usar_M_1=True (padrão): cutoff = último dia do mês anterior (M-1).
      Ex.: se ref_date = 31/03/2024 → cutoff = 29/02/2024.
           Assim, a janela 1m inclui apenas FEV/24.
    - usar_M_1=False: cutoff = a própria data de referência (M).
      Ex.: se ref_date = 31/03/2024 → cutoff = 31/03/2024.
           Assim, a janela 1m inclui MAR/24.

    -------------------------
    Definição das janelas:
    -------------------------
    - Cada janela Xm é formada pelos X meses fechados anteriores ao cutoff.
      Ex.: ref_date = 31/03/2024, usar_M_1=True, janela 3m = dez/23, jan/24, fev/24.
    - "ever": considera todo o histórico do cliente até o cutoff.

    -------------------------
    Variáveis criadas:
    -------------------------
    - tempo_desde_primeira_Xm : dias entre a primeira transação na janela e o cutoff.
                                → Mede a antiguidade da atividade.
    - tempo_desde_ultima_Xm   : dias entre a última transação na janela e o cutoff.
                                → Mede a recência (há quanto tempo o cliente está parado).
    - tempo_atividade_Xm      : diferença entre os dois (primeira – última).
                                → Mede a duração da atividade na janela:
                                   * >0  → várias transações (tempo entre 1ª e última).
                                   * 0   → apenas 1 transação (atividade pontual).
                                   * NaN → nenhuma transação no período.

    -------------------------
    Convenções:
    -------------------------
    - NaN: janela sem transações.
    - 0: houve apenas uma transação no período, ou uma transação exatamente no cutoff.
    - Valores positivos: dias em relação ao cutoff definido.

    -------------------------
    Exemplo prático:
    -------------------------
    Transações de C1:
        2024-03-04
        2024-03-23
        2024-04-22

    ref_date = 30/04/2024, usar_M_1=True → cutoff = 31/03/2024
    - janela 1m = mar/24
    - primeira = 04/03, última = 23/03
    - tempo_desde_primeira_1m = 27 dias (31/03 – 04/03)
    - tempo_desde_ultima_1m   = 8 dias  (31/03 – 23/03)
    - tempo_atividade_1m      = 19 dias (27 – 8)

    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "9m": 9,
               "12m": 12, "24m": 24, "ever": None}
    clientes_com_tx = set(df_tx[id_col].unique())

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]

        # Definição do cutoff
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        # Cliente sem histórico
        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                **{f"tempo_desde_primeira_{k}": np.nan for k in janelas.keys()},
                **{f"tempo_desde_ultima_{k}": np.nan for k in janelas.keys()},
                **{f"tempo_atividade_{k}": np.nan for k in janelas.keys()}
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col] == cid) & (df_tx[dt_col] <= cutoff)]
        feats = {}

        for label, meses in janelas.items():
            if meses is None:  # ever = todo histórico até cutoff
                tx_window = tx_cliente
            else:
                # início da janela = primeiro dia do mês (cutoff - (meses-1) meses)
                start = (cutoff - pd.DateOffset(months=meses-1)).replace(day=1)
                tx_window = tx_cliente[(tx_cliente[dt_col] >= start) & (
                    tx_cliente[dt_col] <= cutoff)]

            if tx_window.empty:
                feats[f"tempo_desde_primeira_{label}"] = np.nan
                feats[f"tempo_desde_ultima_{label}"] = np.nan
                feats[f"tempo_atividade_{label}"] = np.nan
            else:
                primeira = tx_window[dt_col].min()
                ultima = tx_window[dt_col].max()

                t_primeira = (cutoff - primeira).days
                t_ultima = (cutoff - ultima).days

                feats[f"tempo_desde_primeira_{label}"] = t_primeira
                feats[f"tempo_desde_ultima_{label}"] = t_ultima
                feats[f"tempo_atividade_{label}"] = t_primeira - t_ultima

        resultados.append({id_col: cid, ref_col: ref_date, **feats})

    return pd.DataFrame(resultados)
