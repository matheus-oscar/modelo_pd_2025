import pandas as pd
import numpy as np


def features_valor_flex(df_tx: pd.DataFrame,
                        df_inad: pd.DataFrame,
                        id_col="id_cliente",
                        val_col="valor_transacao",
                        dt_col="data_transacao",
                        ref_col="data_referencia",
                        usar_M_1=True) -> pd.DataFrame:
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
                        * NaN → ambos zero
                        * -1 → denominador zero e numerador > 0 (início de atividade)
    - delta_vlr_A_vs_B : diferença absoluta de valores entre períodos vizinhos
    - flag_completo_Xm : indica se a janela X meses está completamente observada (1/0)
    - perc_janela_coberta_Xm : proporção de dias observados na janela X meses
    - flag_cliente_novo : 1 se a primeira transação ocorreu nos últimos 6 meses
    """

    resultados = []
    janelas = {"1m": 1, "3m": 3, "6m": 6, "9m": 9,
               "12m": 12, "24m": 24, "ever": None}
    comparacoes = [("1m", "3m"), ("3m", "6m"), ("6m", "9m"),
                   ("9m", "12m"), ("12m", "24m"), ("24m", "ever")]

    clientes_com_tx = set(df_tx[id_col].unique())
    # primeira transação de cada cliente
    primeira_tx_cliente = df_tx.groupby(id_col)[dt_col].min()

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]

        # cutoff = último dia do mês anterior, se usar_M_1=True
        if usar_M_1:
            cutoff = (ref_date - pd.offsets.MonthEnd(1))
        else:
            cutoff = ref_date

        # caso sem transações
        if cid not in clientes_com_tx:
            resultados.append({
                id_col: cid,
                ref_col: ref_date,
                **{f"vlr_trans_{k}": np.nan for k in janelas.keys()},
                "vlr_trans_ult": np.nan,
                "vlr_trans_max": np.nan,
                "vlr_trans_min": np.nan,
                **{f"comp_vlr_{a}_vs_{b}": np.nan for a, b in comparacoes},
                **{f"delta_vlr_{a}_vs_{b}": np.nan for a, b in comparacoes},
                **{f"flag_completo_{k}": 0 for k in janelas.keys() if k != "ever"},
                **{f"perc_janela_coberta_{k}": 0.0 for k in janelas.keys() if k != "ever"},
                "flag_cliente_novo": np.nan
            })
            continue

        tx_cliente = df_tx[(df_tx[id_col] == cid) & (df_tx[dt_col] <= cutoff)]
        feats = {}

        # Totais por janela (meses fechados)
        for label, meses in janelas.items():
            if meses is None:  # ever
                tx_window = tx_cliente
                feats[f"vlr_trans_{label}"] = tx_window[val_col].sum(
                ) if not tx_window.empty else 0
            else:
                # início da janela = primeiro dia do mês (cutoff - (meses-1) meses)
                start = (cutoff - pd.DateOffset(months=meses-1)).replace(day=1)
                tx_window = tx_cliente[(tx_cliente[dt_col] >= start) & (
                    tx_cliente[dt_col] <= cutoff)]

                feats[f"vlr_trans_{label}"] = tx_window[val_col].sum(
                ) if not tx_window.empty else 0

                # flag completude (janela completa se primeira transação <= start)
                flag_completo = int(primeira_tx_cliente[cid] <= start)
                feats[f"flag_completo_{label}"] = flag_completo

                # percentual de cobertura em dias corridos dentro da janela
                dias_esperados = (cutoff - start).days + 1
                dias_com_historico = (
                    cutoff - max(primeira_tx_cliente[cid], start)).days + 1
                dias_com_historico = max(dias_com_historico, 0)
                feats[f"perc_janela_coberta_{label}"] = round(
                    dias_com_historico / dias_esperados, 3)

        # Última, máxima e mínima
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

        # Comparações vizinhas (regra unificada)
        for a, b in comparacoes:
            v1 = feats[f"vlr_trans_{a}"]
            v2 = feats[f"vlr_trans_{b}"]
            if v1 == 0 and v2 == 0:
                comp = np.nan
            elif v2 == 0:
                comp = -1
            else:
                comp = round(v1/v2, 3)
            feats[f"comp_vlr_{a}_vs_{b}"] = comp
            feats[f"delta_vlr_{a}_vs_{b}"] = v1 - v2

        # Flag cliente novo (entrou nos últimos 6 meses em relação à ref_date)
        feats["flag_cliente_novo"] = int(
            primeira_tx_cliente[cid] > (ref_date - pd.DateOffset(months=6)))

        resultados.append({id_col: cid, ref_col: ref_date, **feats})

    return pd.DataFrame(resultados)
