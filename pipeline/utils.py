import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder
import matplotlib.pyplot as plt


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


def remover_vars(abt, target="atraso_90d", iv_threshold=0.01, corr_threshold=0.8, cols_drop=[]):
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


def plot_categ(df, column):
    # Plotar um barra de contagem de categoria
    aux = df.groupby(column)[column].count().reset_index(name='Qtd.')
    aux.plot.bar(x=column, y='Qtd.', rot=0, figsize=(12, 4))


def plot_txmau_categ(df, column, column_mau, mau=1):
    # usando mau = 1
    df2 = df[[column, column_mau]].copy()
    df2['mau'] = [1 if x == mau else 0 for x in df2[column_mau]]
    aux = df2.groupby(column)["mau"].agg(["mean", 'count'])
    aux = aux.rename(
        columns={'mean': 'Tx. inadimplência', 'count': 'Volumetria'})
    aux.plot.bar(rot=45, subplots=True, figsize=(12, 4), fontsize=8)


def ks(data=None, target=None, prob=None, printar=False, return_ks=False):
    # Calcular KS

    # Copiar a base para não influenciar na base original (data)
    data = data[[target, prob]].copy()

    # target = event; target0 = non_event
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events'] = grouped.sum()[target]
    kstable['non_events'] = grouped.sum()['target0']
    kstable = kstable.sort_values(
        by="min_prob", ascending=False).reset_index(drop=True)
    kstable['event rate'] = (
        kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['non_event rate'] = (
        kstable.non_events / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['event cum_rate'] = (kstable.events / data[target].sum()).cumsum()
    kstable['non_event cum_rate'] = (
        kstable.non_events / data['target0'].sum()).cumsum()
    kstable['KS'] = abs(np.round(kstable['event cum_rate'] *
                        100 - kstable['non_event cum_rate'] * 100, 1))  # * 100

    kstable['n'] = kstable['events'] + kstable['non_events']
    kstable['Class Event Rate'] = np.round(
        kstable.events / kstable.n, 3).apply('{0:.2%}'.format)
    kstable['%'] = (kstable['n'] / kstable['n'].sum()).apply('{0:.0%}'.format)

    # Formating
    kstable['event cum_rate'] = kstable['event cum_rate'].apply(
        '{0:.2%}'.format)
    kstable['non_event cum_rate'] = kstable['non_event cum_rate'].apply(
        '{0:.2%}'.format)
    kstable.index = range(1, 11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 12)
    if printar:
        print(kstable)

    if return_ks:
        return (max(kstable['KS']))

    # Display KS
    from colorama import Fore
    print(Fore.RED +
          "KS is " +
          str(max(kstable['KS'])) + "%" +
          " at decile " +
          str((kstable.index[kstable['KS'] == max(kstable['KS'])][0])))
    return (kstable)


def get_precisions_recalls(actual, preds):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    precision_0 = np.sum((actual == 0) & (preds == 0)) / np.sum(preds == 0)
    precision_1 = np.sum((actual == 1) & (preds == 1)) / np.sum(preds == 1)

    plt.bar([0, 1], [precision_0, precision_1])
    plt.xticks([0, 1], ['Class 0', 'Class 1'], fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.ylabel('Precision', fontsize=20)
    plt.title(
        f'Precision Class 0: {round(precision_0, 2)}\nPrecision Class 1: {round(precision_1, 2)}', fontsize=20)

    plt.subplot(1, 2, 2)
    recall_0 = np.sum((actual == 0) & (preds == 0)) / np.sum(actual == 0)
    recall_1 = np.sum((actual == 1) & (preds == 1)) / np.sum(actual == 1)

    plt.bar([0, 1], [recall_0, recall_1])
    plt.xticks([0, 1], ['Class 0', 'Class 1'], fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.ylabel('Recall', fontsize=20)
    plt.title(
        f'Recall Class 0: {round(recall_0, 2)}\nRecall Class 1: {round(recall_1, 2)}', fontsize=20)

    plt.tight_layout()
    plt.show()


def taxa_inadimplencia_por_variavel(df, var, target="atraso_90d", bins=10):
    """
    Calcula a taxa de inadimplência por faixas de uma variável contínua.

    Retorna colunas:
    faixa | n | n_bons | n_maus | taxa_inadimplencia
    """
    df = df.copy()

    # Bins dinâmicos (quantis) ou lista fixa
    if isinstance(bins, int):
        df["faixa"] = pd.qcut(df[var], q=bins, duplicates="drop")
    else:
        df["faixa"] = pd.cut(df[var], bins=bins, include_lowest=True)

    taxa = (
        df.groupby("faixa")
        .agg(
            n=(target, "size"),
            n_bons=(target, lambda x: (x == 0).sum()),
            n_maus=(target, lambda x: (x == 1).sum()),
            taxa_inadimplencia=(target, "mean")
        )
        .reset_index()
    )
    return taxa


def plot_inad_var(df, var, target="atraso_90d", bins=10):
    """
    Plota taxa de inadimplência e taxa de bons por faixas de uma variável contínua.
    Mostra apenas o percentual de inadimplência em cada ponto.
    """
    taxa = taxa_inadimplencia_por_variavel(df, var, target, bins)

    plt.figure(figsize=(10, 5))
    plt.plot(taxa["faixa"].astype(str), taxa["taxa_inadimplencia"],
             marker="o", label="Taxa de inadimplência")
    plt.plot(taxa["faixa"].astype(str), 1 - taxa["taxa_inadimplencia"],
             marker="s", linestyle="--", label="Taxa de bons")

    plt.xticks(rotation=45)
    plt.ylabel("Taxa")
    plt.title(f"Taxas por {var}")

    # Adiciona apenas o percentual da taxa de inadimplência
    for i, row in taxa.iterrows():
        pct = f"{row['taxa_inadimplencia']*100:.1f}%"
        plt.text(i, row["taxa_inadimplencia"]+0.02, pct,
                 ha="center", fontsize=8, color="black")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return taxa
