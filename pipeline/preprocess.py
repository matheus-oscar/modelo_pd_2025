import os
import pandas as pd
import numpy as np

# -------------------------------------
# CLIENTES
# -------------------------------------


def preprocessar_clientes(
    df_cli: pd.DataFrame,
    id_col="id_cliente",
    idade_col="idade",
    renda_col="renda_mensal",
    dt_abertura_col="data_abertura_conta",
    score_col="score_interno",
):
    """
    Etapas de pré-processamento da base de clientes:
    - Conversão de datas para os formatos apropriados
    - Conversão de idade, renda e score para numérico.
    - Criação da coluna mes_abertura_conta em 'YYYY-MM'.
    - Normaliza coluna *estado_civil* (minúsculo, sem espaços).
    """

    df = df_cli.copy()

    if dt_abertura_col in df.columns:
        df[dt_abertura_col] = pd.to_datetime(
            df[dt_abertura_col], format="%d/%m/%Y", errors="coerce"
        )

    if idade_col in df.columns:
        df[idade_col] = pd.to_numeric(df[idade_col], errors="coerce")
    if renda_col in df.columns:
        df[renda_col] = pd.to_numeric(df[renda_col], errors="coerce")
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    if dt_abertura_col in df.columns:
        df["mes_abertura_conta"] = df[dt_abertura_col].dt.to_period(
            "M").astype(str)

    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str)

    if "estado_civil" in df.columns:
        df["estado_civil"] = (
            df["estado_civil"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    colunas = [
        id_col, idade_col, renda_col, dt_abertura_col, "mes_abertura_conta",
        "estado_civil", "tempo_emprego_anos", "qtde_produtos",
        score_col, "limite_credito"
    ]

    return df[colunas]

# -------------------------------------
# INADIMPLÊNCIA
# -------------------------------------


def preprocessar_inadimplencia(
    df_inad: pd.DataFrame,
    id_col="id_cliente",
    mes_col="mes_safra",
    perf="atraso_90d"
):
    """
    - Converte mes_safra no formato 'YYYY-MM'  
    - Cria data_referencia como o último dia do mês relativo à coluna *mes_safra*.
    - Normaliza o target apenas onde não é nulo, corrigindo os casos em que *atraso_90d* tem valor igual a 5.
    """

    df = df_inad.copy()

    df[id_col] = df[id_col].astype(str)

    df[mes_col] = pd.to_datetime(df[mes_col], format="%Y-%m", errors="raise")

    df["data_referencia"] = df[mes_col] + pd.offsets.MonthEnd(0)

    df[mes_col] = df[mes_col].dt.to_period("M").astype(str)

    df[perf] = pd.to_numeric(df[perf], errors="coerce")

    mask = df[perf].notna()

    df.loc[mask, perf] = np.where(df.loc[mask, perf] >= 1, 1, 0).astype(int)

    

    colunas = [id_col, mes_col, "data_referencia", perf]

    df = df[colunas]

    return df

# -------------------------------------
# TRANSAÇÕES
# -------------------------------------


def preprocessar_transacoes(
    df_tx: pd.DataFrame,
    id_col="id_cliente",
    val_col="valor_transacao",
    dt_col="data_transacao"
):
    """
    - Converte data_transacao para o formato 'YYYY-MM-DD'  .
    - Cria a coluna *mes_safra* no formato 'YYYY-MM' a partir de *data_transacao*.
    - Ordena por id_cliente, data_transacao.
    """
    df = df_tx.copy()

    df[id_col] = df[id_col].astype(str)

    df[dt_col] = pd.to_datetime(df[dt_col], format="%d/%m/%Y", errors="coerce")

    df["mes_safra"] = df[dt_col].dt.to_period("M").astype(str)

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    colunas = [id_col, dt_col, "mes_safra", val_col]

    df = df[colunas]

    return df
