# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# Função auxiliar - binning inicial
# ==========================================
def binarizar_var(df, var, target, q=10):
    """
    Cria faixas iniciais para uma variável contínua (quantis + nulos).
    """
    df_aux = df.copy()
    # Binning apenas para não-nulos
    df_aux["faixa"] = pd.qcut(df_aux[var].dropna(), q=q, duplicates="drop")
    df_aux["faixa"] = df_aux["faixa"].astype(str)
    # Nulos como -99
    df_aux.loc[df_aux[var].isna(), "faixa"] = "-99"
    return df_aux


# ==========================================
# Função auxiliar - tabela consolidada
# ==========================================
def tabela_consolidada(df, var_cat, target):
    """
    Retorna tabela com categorias + métricas (N, taxas, WOE, IV, IV total).
    """
    tab = (
        df.groupby(var_cat)[target]
        .agg(N_total="count", N_maus="sum")
        .reset_index()
    )
    tab["N_bons"] = tab["N_total"] - tab["N_maus"]
    tab["tx_default"] = tab["N_maus"] / tab["N_total"]
    tab["tx_n_default"] = tab["N_bons"] / tab["N_total"]

    # Proporções
    total_bons = tab["N_bons"].sum()
    total_maus = tab["N_maus"].sum()
    tab["P_bons"] = tab["N_bons"] / total_bons
    tab["P_maus"] = tab["N_maus"] / total_maus

    # WOE e IV
    def calc_woe(p_b, p_m):
        if p_b > 0 and p_m > 0:
            return math.log(p_b / p_m)
        else:
            return 0

    tab["WOE"] = tab.apply(lambda r: calc_woe(r["P_bons"], r["P_maus"]), axis=1)
    tab["IV"] = (tab["P_bons"] - tab["P_maus"]) * tab["WOE"]

    iv_total = tab["IV"].sum()

    # Linha de totalização
    total_row = pd.DataFrame([{
        var_cat: "TOTAL",
        "N_total": tab["N_total"].sum(),
        "N_bons": tab["N_bons"].sum(),
        "N_maus": tab["N_maus"].sum(),
        "tx_default": tab["N_maus"].sum() / tab["N_total"].sum(),
        "tx_n_default": tab["N_bons"].sum() / tab["N_total"].sum(),
        "WOE": np.nan,
        "IV": iv_total
    }])

    tab = pd.concat([tab, total_row], ignore_index=True)
    return tab


# ==========================================
# Função auxiliar - gráfico por safra
# ==========================================
def plot_taxa_por_safra(df, var_cat, target, safra_col="mes_safra"):
    resumo = (
        df.groupby([safra_col, var_cat])[target]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    for cat in resumo[var_cat].unique():
        subset = resumo[resumo[var_cat] == cat]
        plt.plot(subset[safra_col], subset[target], marker="o", label=str(cat))
    plt.title(f"Taxa de Default por Safra ({var_cat})")
    plt.xlabel("Safra")
    plt.ylabel("Taxa de Default")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)


# ==========================================
# Streamlit App
# ==========================================
st.title("🔎 App de Binning e Reagrupamento Interativo")

# Upload de base
file = st.file_uploader("Carregue sua base (CSV)", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.write("✅ Base carregada:", df.shape)

    # Selecionar variável
    var = st.selectbox("Selecione a variável para análise", df.columns)
    target = st.selectbox("Selecione a variável target (0/1)", df.columns)
    safra_col = st.selectbox("Selecione a variável de safra", df.columns)

    if var and target and safra_col:
        # Binning inicial
        df_aux = binarizar_var(df, var, target)

        # Mostrar tabela consolidada inicial
        st.subheader("📊 Tabela consolidada (auto-binning inicial)")
        tab_ini = tabela_consolidada(df_aux, "faixa", target)
        st.dataframe(tab_ini)

        # Reagrupamento manual
        st.subheader("✏️ Reagrupamento manual")
        categorias = df_aux["faixa"].unique()
        grupos = {}
        for cat in categorias:
            novo_grupo = st.text_input(f"Defina grupo para {cat}", value=cat)
            grupos[cat] = novo_grupo

        # Aplicar reagrupamento
        df_aux["faixa_final"] = df_aux["faixa"].map(grupos)

        # Mostrar tabela consolidada final
        st.subheader("📊 Tabela consolidada (após reagrupamento)")
        tab_final = tabela_consolidada(df_aux, "faixa_final", target)
        st.dataframe(tab_final)

        # Gráfico
        st.subheader("📈 Taxa de default por safra")
        plot_taxa_por_safra(df_aux, "faixa_final", target, safra_col)

        # Exportação
        st.subheader("💾 Exportar base recategorizada")
        if st.button("Gerar base final"):
            df_export = df.copy()
            df_export[f"{var}_cat"] = df_aux["faixa_final"]
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download base_final.csv",
                data=csv,
                file_name="base_final.csv",
                mime="text/csv",
            )
