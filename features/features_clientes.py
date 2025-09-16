import pandas as pd
import numpy as np


def features_clientes(df_cli: pd.DataFrame,
                      df_inad: pd.DataFrame,
                      id_col: str = "id_cliente",
                      dt_abertura_col: str = "data_abertura_conta",
                      idade_col: str = "idade",
                      renda_col: str = "renda_mensal",
                      limite_col: str = "limite_credito",
                      qtde_prod_col: str = "qtde_produtos",
                      ref_col: str = "data_referencia",
                      usar_M_1: bool = True) -> pd.DataFrame:
    """
    Gera features de clientes combinando atributos cadastrais (originais) e derivadas.

    Alinhamento temporal:
    ---------------------
    - usar_M_1=True (padrão): cutoff = último dia do mês anterior (M-1).
      Ex.: ref_date=30/04/2024 → cutoff=31/03/2024.
    - usar_M_1=False: cutoff = a própria data de referência (M).

    Saída:
    ------
    DataFrame expandido por cliente e safra (data_referencia), contendo:
    - Todas as colunas originais de df_cli.
    - Novas features derivadas:
        * tempo_relacionamento_anos
        * idade2
        * log_renda
        * renda_por_limite
    """

    cli_index = df_cli.set_index(id_col)
    registros = []

    for _, row in df_inad.iterrows():
        cid = row[id_col]
        ref_date = row[ref_col]
        cutoff = (ref_date - pd.offsets.MonthEnd(1)) if usar_M_1 else ref_date

        if cid not in cli_index.index:
            # Preenche com NaN para todas as colunas originais + derivadas
            vazio = {col: np.nan for col in df_cli.columns if col != id_col}
            vazio.update({
                id_col: cid,
                ref_col: ref_date,
                "tempo_relacionamento_anos": np.nan,
                "idade2": np.nan,
                "log_renda": np.nan,
                "renda_por_limite": np.nan
            })
            registros.append(vazio)
            continue

        cli = cli_index.loc[cid].to_dict()

        # tempo de relacionamento (anos)
        dt_abertura = cli.get(dt_abertura_col, pd.NaT)
        if pd.isna(dt_abertura) or (cutoff <= dt_abertura):
            anos_rel = np.nan
        else:
            anos_rel = round((cutoff - dt_abertura).days / 365.25, 4)

        # features derivadas
        idade2 = cli[idade_col] ** 2 if not pd.isna(
            cli.get(idade_col)) else np.nan
        log_renda = np.log1p(cli.get(renda_col, np.nan))
        renda_por_limite = (
            cli[renda_col] / cli[limite_col]
            if (not pd.isna(cli.get(renda_col)) and cli.get(limite_col, 0) > 0)
            else np.nan
        )

        # montar registro
        registro = cli.copy()
        registro.update({
            id_col: cid,
            ref_col: ref_date,
            "tempo_relacionamento_anos": anos_rel,
            "idade2": idade2,
            "log_renda": log_renda,
            "renda_por_limite": renda_por_limite
        })
        registros.append(registro)

    return pd.DataFrame(registros).sort_values([id_col, ref_col]).reset_index(drop=True)
