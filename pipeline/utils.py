import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder


def diagnostico_abt(abt, target="atraso_90d"):
    resumo = []

    for col in abt.columns:
        if col in [target] + cols_drop:
            continue

        serie = abt[col]
        tipo = str(serie.dtype)

        # % missing
        pct_missing = serie.isna().mean() * 100

        # Cardinalidade (categorias ou valores únicos)
        n_unique = serie.nunique(dropna=True)

        # Estatísticas básicas só para numéricas
        if pd.api.types.is_numeric_dtype(serie):
            desc = serie.describe()
            minimo = desc["min"]
            maximo = desc["max"]
            media = desc["mean"]
        else:
            minimo = maximo = media = None

        resumo.append({
            "variavel": col,
            "tipo": tipo,
            "pct_missing": round(pct_missing, 2),
            "n_unique": n_unique,
            "media": media,
            "min": minimo,
            "max": maximo
        })

    return pd.DataFrame(resumo).sort_values("pct_missing", ascending=False)


def aplicar_ohe_completo(abt, excluir=[]):
    """
    Aplica One-Hot Encoding com feature_engine,
    mantendo também as colunas originais.

    Parâmetros
    ----------
    abt : DataFrame
        Base de dados (ABT).
    excluir : list, opcional
        Lista de colunas que NÃO devem ser processadas.
    """
    excluir = excluir or []

    # Detectar categóricas candidatas
    cat_features = [
        col for col in abt.columns
        if abt[col].dtype.name in ["object", "category"] and col not in excluir
    ]

    if not cat_features:
        return abt.copy()

    encoder = OneHotEncoder(
        variables=cat_features,
        drop_last=False,
        ignore_format=True
    )

    abt_ohe = encoder.fit_transform(abt)

    return abt_ohe


def calcular_iv(df, feature, target, bins=10):
    # binarização: só para numéricas contínuas
    if pd.api.types.is_numeric_dtype(df[feature]):
        try:
            df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
        except ValueError:
            df["bin"] = df[feature]
    else:
        df["bin"] = df[feature]

    tab = pd.crosstab(df["bin"], df[target])
    tab = tab.apply(lambda x: x/x.sum(), axis=0)  # distribuição por classe
    tab["woe"] = np.log((tab[1] + 1e-6) / (tab[0] + 1e-6))
    tab["iv"] = (tab[1] - tab[0]) * tab["woe"]

    return tab["iv"].sum()


def avaliar_iv(abt, target="atraso_90d", top=20):
    ivs = {}
    for col in abt.columns:
        if col in [target] + cols_drop + ['estado_civil']:
            continue
        try:
            ivs[col] = calcular_iv(abt[[col, target]].dropna(), col, target)
        except Exception as e:
            ivs[col] = np.nan
    df_iv = pd.DataFrame.from_dict(ivs, orient="index", columns=["IV"])
    return df_iv.sort_values("IV", ascending=False).head(top)


def calcular_iv_woe(df, feature, target="atraso_90d", bins=10):
    """
    Calcula IV (Information Value) e WOE para uma variável.
    Retorna IV e tabela com bins + WOE.
    """
    temp = df[[feature, target]].copy().dropna()
    if temp.empty:
        return np.nan, pd.DataFrame()

    try:
        temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")
    except Exception:
        temp["bin"] = pd.cut(temp[feature], bins=bins)

    grouped = temp.groupby("bin")[target].agg(["count", "sum"])
    grouped = grouped.rename(columns={"count": "total", "sum": "bad"})
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    grouped["dist_good"] = grouped["good"] / total_good
    grouped["dist_bad"] = grouped["bad"] / total_bad
    grouped["woe"] = np.log(
        (grouped["dist_good"] + 1e-6) / (grouped["dist_bad"] + 1e-6))
    grouped["iv"] = (grouped["dist_good"] -
                     grouped["dist_bad"]) * grouped["woe"]

    iv = grouped["iv"].sum()
    return iv, grouped


def calcular_iv(df, feature, target, bins=10):
    if pd.api.types.is_numeric_dtype(df[feature]):
        try:
            df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
        except Exception:
            df["bin"] = df[feature]
    else:
        df["bin"] = df[feature]

    tab = pd.crosstab(df["bin"], df[target])
    tab = tab.apply(lambda x: x / x.sum(), axis=0)  # distribuição %
    tab["woe"] = np.log((tab[1] + 1e-6) / (tab[0] + 1e-6))
    tab["iv"] = (tab[1] - tab[0]) * tab["woe"]

    return tab["iv"].sum()


def comparar_iv(df_M, df_M1, feature, target="atraso_90d", bins=10):
    """
    Compara IV para uma mesma feature entre df_M e df_M1.
    Retorna dicionário com IVs e diferença percentual.
    """
    iv_M, _ = calcular_iv_woe(df_M, feature, target, bins)
    iv_M1, _ = calcular_iv_woe(df_M1, feature, target, bins)

    return {
        "variavel": feature,
        "IV_M": iv_M,
        "IV_M1": iv_M1,
        "delta_IV": None if (iv_M is None or iv_M == 0) else round((iv_M1 - iv_M) / iv_M, 3)
    }


def calcular_ks(df, feature, target="atraso_90d", bins=10):
    """
    Calcula KS para uma variável.
    """
    temp = df[[feature, target]].copy().dropna()
    if temp.empty:
        return np.nan

    try:
        temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")
    except Exception:
        temp["bin"] = pd.cut(temp[feature], bins=bins)

    grouped = temp.groupby("bin")[target].agg(["count", "sum"])
    grouped = grouped.rename(columns={"count": "total", "sum": "bad"})
    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["cum_bad"] = grouped["bad"].cumsum() / grouped["bad"].sum()
    grouped["cum_good"] = grouped["good"].cumsum() / grouped["good"].sum()

    ks = np.max(np.abs(grouped["cum_bad"] - grouped["cum_good"]))
    return ks


def comparar_ks(df_M, df_M1, feature, target="atraso_90d", bins=10):
    """
    Compara KS para uma mesma feature entre df_M e df_M1.
    Retorna dicionário com KSs e diferença percentual.
    """
    ks_M = calcular_ks(df_M, feature, target, bins)
    ks_M1 = calcular_ks(df_M1, feature, target, bins)

    return {
        "variavel": feature,
        "KS_M": ks_M,
        "KS_M1": ks_M1,
        "delta_KS": None if (ks_M is None or ks_M == 0) else round((ks_M1 - ks_M) / ks_M, 3)
    }


def remover_vars(abt, target="atraso_90d", iv_threshold=0.01, corr_threshold=0.8):
    ivs = {}
    for col in abt.columns:
        if col in [target, 'estado_civil'] + cols_drop:
            continue
        try:
            ivs[col] = calcular_iv(abt[[col, target]].dropna(), col, target)
        except Exception:
            ivs[col] = np.nan

    df_iv = pd.DataFrame.from_dict(
        ivs, orient="index", columns=["IV"]).dropna()
    df_iv = df_iv.sort_values("IV", ascending=False)

    # Seleciona só as variáveis acima do limiar
    selecionadas = df_iv[df_iv["IV"] >= iv_threshold].index.tolist()

    # --- Remover correlação alta ---
    df_corr = abt[selecionadas].corr().abs()
    to_drop = set()
    for i in df_corr.columns:
        for j in df_corr.columns:
            if i != j and df_corr.loc[i, j] > corr_threshold:
                # Mantém a de maior IV
                if df_iv.loc[i, "IV"] >= df_iv.loc[j, "IV"]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)

    finais = [v for v in selecionadas if v not in to_drop]

    return {
        "iv_ranking": df_iv,
        "selecionadas_iniciais": selecionadas,
        "removidas_corr": list(to_drop),
        "final": finais
    }


def cutoff_otimo_ks(y_true, y_pred_proba):
    df = pd.DataFrame({"y": y_true, "score": y_pred_proba})
    df = df.sort_values("score", ascending=True)
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["ks"] = abs(df["cum_bad"] - df["cum_good"])
    idx_max = df["ks"].idxmax()
    return df.loc[idx_max, "score"], df.loc[idx_max, "ks"]


def plotar_ks(y_true, y_pred_proba, titulo="KS Curve"):
    # Construir DataFrame
    df = pd.DataFrame({"y": y_true, "score": y_pred_proba})
    df = df.sort_values("score", ascending=True)

    # Acumulados
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["ks"] = abs(df["cum_bad"] - df["cum_good"])

    # Ponto de KS máximo
    idx_max = df["ks"].idxmax()
    ks_val = df.loc[idx_max, "ks"]
    score_ks = df.loc[idx_max, "score"]

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(df["score"], df["cum_good"], label="Bons acumulados (y=0)")
    plt.plot(df["score"], df["cum_bad"], label="Maus acumulados (y=1)")
    # plt.vlines(x=score_ks, ymin=df.loc[idx_max,"cum_good"], ymax=df.loc[idx_max,"cum_bad"],
    #            colors="red", linestyles="--", label=f"KS={ks_val:.3f} @ cutoff {score_ks:.3f}")
    plt.title(titulo)
    plt.xlabel("Probabilidade de Default")
    plt.ylabel("Proporção acumulada")
    plt.legend()
    plt.show()

    return ks_val, score_ks
