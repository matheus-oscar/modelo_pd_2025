import pandas as pd
import numpy as np


def features_clientes_transacional(df_cli: pd.DataFrame,
                                   df_inad: pd.DataFrame,
                                   id_col: str = "id_cliente",
                                   dt_abertura_col: str = "data_abertura_conta",
                                   ref_col: str = "data_referencia",
                                   usar_M_1: bool = False) -> pd.DataFrame:
    """
    Gera variáveis cadastrais dependentes da data de referência.

    Variáveis criadas
    -----------------
    - tempo_relacionamento_anos / meses
    - flag_cliente_antigo (1 se >5 anos, 0 se não, NaN se não disponível)
    - idade_relativa
    - limite_por_ano_conta, limite_por_mes_conta
    - produtos_por_ano_conta, produtos_por_mes_conta

    Flags auxiliares
    -----------------
    - tempo_relacionamento_isna: 1 se não há data_abertura ou se inválida
    - flag_cliente_antigo_isna : 1 se não foi possível calcular
    - *_isna                   : 1 se razões não puderam ser calculadas
    """

    cli_index = df_cli.set_index(id_col)
    registros = []

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in cli_index.index:
            registros.append({
                id_col: cid,
                ref_col: ref_date,
                "tempo_relacionamento_anos": np.nan,
                "tempo_relacionamento_meses": np.nan,
                "tempo_relacionamento_isna": 1,
                "flag_cliente_antigo": np.nan,
                "flag_cliente_antigo_isna": 1,
                "idade_relativa": np.nan,
                "limite_por_ano_conta": np.nan,
                "limite_por_ano_conta_isna": 1,
                "limite_por_mes_conta": np.nan,
                "limite_por_mes_conta_isna": 1,
                "produtos_por_ano_conta": np.nan,
                "produtos_por_ano_conta_isna": 1,
                "produtos_por_mes_conta": np.nan,
                "produtos_por_mes_conta_isna": 1,
            })
            continue

        cli = cli_index.loc[cid]
        dt_abertura = cli.get(dt_abertura_col, pd.NaT)

        # tempos
        if pd.isna(dt_abertura):
            anos_rel = np.nan
            meses_rel = np.nan
            tempo_isna = 1
        else:
            delta_days = (cutoff - dt_abertura).days
            if pd.isna(delta_days) or delta_days <= 0:
                anos_rel = np.nan
                meses_rel = np.nan
                tempo_isna = 1
            else:
                anos_rel = round(delta_days / 365.25, 4)
                meses_rel = round(delta_days / 30.44, 4)
                tempo_isna = 0

        # flag cliente antigo
        if pd.isna(anos_rel):
            flag_antigo = np.nan
            flag_antigo_isna = 1
        else:
            flag_antigo = 1 if anos_rel > 5 else 0
            flag_antigo_isna = 0

        # idade relativa
        idade_rel = np.nan
        if "idade" in cli.index and not pd.isna(cli["idade"]) and not pd.isna(anos_rel):
            idade_rel = round(float(cli["idade"]) - float(anos_rel), 4)

        # razões
        limite = cli.get("limite_credito", np.nan)
        qtd_prod = cli.get("qtde_produtos", np.nan)

        lim_por_ano = np.nan
        lim_por_mes = np.nan
        prod_por_ano = np.nan
        prod_por_mes = np.nan
        lim_ano_isna = lim_mes_isna = prod_ano_isna = prod_mes_isna = 1

        if not pd.isna(anos_rel) and anos_rel > 0:
            lim_por_ano = limite / anos_rel if not pd.isna(limite) else np.nan
            prod_por_ano = qtd_prod / \
                anos_rel if not pd.isna(qtd_prod) else np.nan
            lim_ano_isna = 0 if not pd.isna(lim_por_ano) else 1
            prod_ano_isna = 0 if not pd.isna(prod_por_ano) else 1

        if not pd.isna(meses_rel) and meses_rel > 0:
            lim_por_mes = limite / meses_rel if not pd.isna(limite) else np.nan
            prod_por_mes = qtd_prod / \
                meses_rel if not pd.isna(qtd_prod) else np.nan
            lim_mes_isna = 0 if not pd.isna(lim_por_mes) else 1
            prod_mes_isna = 0 if not pd.isna(prod_por_mes) else 1

        registros.append({
            id_col: cid,
            ref_col: ref_date,
            "tempo_relacionamento_anos": anos_rel,
            "tempo_relacionamento_meses": meses_rel,
            "tempo_relacionamento_isna": tempo_isna,
            "flag_cliente_antigo": flag_antigo,
            "flag_cliente_antigo_isna": flag_antigo_isna,
            "idade_relativa": idade_rel,
            "limite_por_ano_conta": lim_por_ano,
            "limite_por_ano_conta_isna": lim_ano_isna,
            "limite_por_mes_conta": lim_por_mes,
            "limite_por_mes_conta_isna": lim_mes_isna,
            "produtos_por_ano_conta": prod_por_ano,
            "produtos_por_ano_conta_isna": prod_ano_isna,
            "produtos_por_mes_conta": prod_por_mes,
            "produtos_por_mes_conta_isna": prod_mes_isna,
        })

    return pd.DataFrame(registros).sort_values([id_col, ref_col]).reset_index(drop=True)
