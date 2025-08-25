import pandas as pd
import numpy as np

# -------------------------------------
# FEATURE ENGINEERING - CLIENTES
# -------------------------------------


def features_cadastrais(df_cli: pd.DataFrame,
                        id_col: str = "id_cliente",
                        idade_col: str = "idade",
                        renda_col: str = "renda_mensal",
                        score_col: str = "score_interno",
                        limite_col: str = "limite_credito",
                        qtde_prod_col: str = "qtde_produtos") -> pd.DataFrame:
    """
    Cria features derivadas da base de clientes 

    Parâmetros
    ----------
    df_cli : DataFrame
        Base de clientes já sanitizada (vinda de preprocess_clientes).
    id_col : str
        Nome da coluna identificadora do cliente.
    dt_abertura_col : str
        Nome da coluna de data de abertura da conta.
    idade_col : str
        Nome da coluna com a idade (anos).
    renda_col : str
        Nome da coluna com a renda mensal.
    score_col : str
        Nome da coluna com o score interno.
    limite_col : str
        Nome da coluna com o limite de crédito.
    qtde_prod_col : str
        Nome da coluna com a quantidade de produtos.
    emprego_col : str
        Nome da coluna com o tempo de emprego (anos).

    Categorias de features
    ----------------------
    - Idade: quadrática
    - Renda: log, relação com limite
    - Relacionamento: tempo de conta, flag cliente antigo
    - Emprego: faixas de estabilidade, razão emprego/relacionamento
    - Produtos: flag de múltiplos produtos
    - Score: flag missing
    
    """

    df = df_cli.copy()

    # -------------------------------
    # Idade
    # -------------------------------
    df["idade2"] = df[idade_col] ** 2
    
    # -------------------------------
    # Renda, renda x limite
    # -------------------------------
    df["log_renda"] = np.log1p(df[renda_col])

    
    df["renda_por_limite"] = np.where(df[limite_col] > 0,
                                      df[renda_col] /  df[limite_col],
                                    np.nan) 

    # -------------------------------
    # Produtos
    # -------------------------------
    df["multi_produto_flag"] = (df[qtde_prod_col] > 1).astype(int)

    # # -------------------------------
    # # Score interno
    # # -------------------------------
    # df["score_interno_na"] = df[score_col].isna().astype(int)


    return df
