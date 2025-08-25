import pandas as pd

from features.features_clientes import features_cadastrais
from features.features_valor import features_valor_flex
from features.features_quantidade import features_quantidade_flex
from features.features_tempo import features_tempo_flex
from features.features_flags import features_flags_flex
from features.features_clientes_transacional import features_clientes_transacional


def gerar_abt(df_clientes, df_inad, df_tx, usar_M_1=True):
    """
    Consolida a ABT (Analytical Base Table) com todas as features.

    Parâmetros
    ----------
    df_clientes : DataFrame preprocessado de clientes
    df_inad : DataFrame preprocessado de inadimplência
    df_tx : DataFrame preprocessado de transações
    usar_M_1 : bool
        Define se cutoff das transações considera fim do próprio mês (False) ou mês anterior (True).

    Retorna
    -------
    DataFrame consolidado (ABT).
    """

    abt = df_inad.copy()

    # 1. Features cadastrais (estáticas)
    feats_cli = features_cadastrais(df_clientes)
    abt = abt.merge(feats_cli, on="id_cliente", how="left")

    # 2. Features de valor
    feats_val = features_valor_flex(df_tx, df_inad, usar_M_1=usar_M_1)
    abt = abt.merge(feats_val, on=["id_cliente",
                    "data_referencia"], how="left")

    # 3. Features de quantidade
    feats_qtd = features_quantidade_flex(df_tx, df_inad, usar_M_1=usar_M_1)
    abt = abt.merge(feats_qtd, on=["id_cliente",
                    "data_referencia"], how="left")

    # 4. Features de tempo
    feats_tmp = features_tempo_flex(df_tx, df_inad, usar_M_1=usar_M_1)
    abt = abt.merge(feats_tmp, on=["id_cliente",
                    "data_referencia"], how="left")

    # 5. Flags
    feats_flags = features_flags_flex(df_tx, df_inad, usar_M_1=usar_M_1)
    abt = abt.merge(feats_flags, on=[
                    "id_cliente", "data_referencia"], how="left")

    # 6. Features de clientes dependentes da referência
    feats_cli_tx = features_clientes_transacional(
        df_clientes, df_inad, usar_M_1=usar_M_1)
    abt = abt.merge(feats_cli_tx, on=[
                    "id_cliente", "data_referencia"], how="left")

    return abt
