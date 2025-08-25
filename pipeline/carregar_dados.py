import pandas as pd

def carregar_dados(diretorio_dados="../data/raw"):
    """
    Lê os dados originais enviados para a resolução do case e retorna em formato de dicionário.

    Dados disponíveis:
      - clientes_case.csv
      - inadimplencia_case.csv
      - transacoes_case.csv
    """
    df_clientes = pd.read_csv(f"{diretorio_dados}/clientes_case.csv", sep=";")
    df_inadimplencia = pd.read_csv(
        f"{diretorio_dados}/inadimplencia_case.csv", sep=";")
    df_transacoes = pd.read_csv(
        f"{diretorio_dados}/transacoes_case.csv", sep=";")

    return {
        "clientes": df_clientes,
        "inadimplencia": df_inadimplencia,
        "transacoes": df_transacoes
    }
