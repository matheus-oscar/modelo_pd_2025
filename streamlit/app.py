import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 App para Categorização", layout="wide")

# ==========================================
# Funções auxiliares
# ==========================================
def binarizar_var(df, var, target, q=5):
    df_aux = df.copy()
    df_aux["faixa"] = pd.qcut(df_aux[var].dropna(), q=q, duplicates="drop")
    df_aux["faixa"] = df_aux["faixa"].astype(str)
    df_aux.loc[df_aux[var].isna(), "faixa"] = "-99"
    return df_aux

def tabela_consolidada(df, var_cat, target):
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

    def calc_woe(p_b, p_m):
        if p_b > 0 and p_m > 0:
            return math.log(p_b / p_m)
        else:
            return 0

    tab["WOE"] = tab.apply(lambda r: calc_woe(r["P_bons"], r["P_maus"]), axis=1)
    tab["IV"] = (tab["P_bons"] - tab["P_maus"]) * tab["WOE"]

    iv_total = tab["IV"].sum()
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
    return pd.concat([tab, total_row], ignore_index=True)

def plot_taxa_por_safra(df, var_cat, target, safra_col="safra"):
    resumo = df.groupby([safra_col, var_cat])[target].mean().reset_index()
    plt.figure(figsize=(8, 4))
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
# App principal
# ==========================================
st.title("📊 App para Categorização")

file = st.sidebar.file_uploader("📂 Carregue sua base (CSV)", type="csv")

# Inicializar session_state
if "resultados" not in st.session_state:
    st.session_state["resultados"] = {}
if "ivs" not in st.session_state:
    st.session_state["ivs"] = {}

if file is not None:
    df = pd.read_csv(file)
    st.success(f"✅ Base carregada: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # ======================
    # Configurações no sidebar
    # ======================
    st.sidebar.header("⚙️ Configurações principais")
    target = st.sidebar.selectbox("Selecione a coluna de target", df.columns)
    safra_col = st.sidebar.selectbox("Selecione a coluna de safra (ou deixe em branco)", [""] + df.columns.tolist())

    valores_unicos = df[target].dropna().unique()
    if len(valores_unicos) > 2:
        st.error(f"A coluna '{target}' tem mais de 2 valores únicos. Selecione uma coluna binária (0/1).")
        st.stop()

    # Criar safra se não foi selecionada
    data_col = None
    if safra_col == "":
        st.sidebar.info("Nenhuma safra escolhida → selecione uma coluna de data")
        data_col = st.sidebar.selectbox("Selecione a coluna de data", df.columns)
        try:
            df[data_col] = pd.to_datetime(df[data_col], errors="coerce")
            df["safra"] = df[data_col].dt.to_period("M").astype(str)
            safra_col = "safra"
            st.success(f"✅ Safra criada a partir de '{data_col}'")
        except Exception as e:
            st.error(f"Erro ao converter coluna {data_col}: {e}")
            st.stop()

    # Seleção de colunas
    st.sidebar.header("⚙️ Pré-processamento")
    excluidas = [target, safra_col] + ([data_col] if data_col else [])
    todas_cols = [c for c in df.columns if c not in excluidas]
    with st.sidebar.expander("Colunas para descartar"):
        cols_drop = st.multiselect("Selecione as colunas que NÃO serão categorizadas", todas_cols, default=[])
    variaveis_analise = [c for c in todas_cols if c not in cols_drop]
    var = st.sidebar.selectbox("Selecione a variável para análise", variaveis_analise)

    # ======================
    # Área principal - Resultados
    # ======================
    if var:
        df_aux = binarizar_var(df, var, target)

        st.subheader("📊 Tabela consolidada (categorização inicial)")
        tab_ini = tabela_consolidada(df_aux, "faixa", target)
        st.dataframe(tab_ini)

        st.subheader("✏️ Reagrupamento manual")
        categorias = tab_ini.loc[tab_ini["faixa"] != "TOTAL", "faixa"].tolist()
        grupos = {cat: st.text_input(f"Grupo para {cat}", value=cat, key=f"{var}_{cat}") for cat in categorias}
        df_aux["faixa_final"] = df_aux["faixa"].map(grupos)

        st.subheader("📊 Tabela consolidada (após reagrupamento)")
        tab_final = tabela_consolidada(df_aux, "faixa_final", target)
        st.dataframe(tab_final)

        st.subheader("📈 Taxa de default por safra")
        plot_taxa_por_safra(df_aux, "faixa_final", target, safra_col)

        # Ações
        st.sidebar.header("💾 Opções")
        if st.sidebar.button("Salvar recategorização"):
            st.session_state["resultados"][var] = df_aux["faixa_final"].copy()
            st.session_state["ivs"][var] = tab_final.loc[tab_final["faixa_final"] == "TOTAL", "IV"].values[0]
            st.success(f"Recategorização de {var} salva!")

        if st.sidebar.button("♻️ Resetar variável atual"):
            st.session_state["resultados"].pop(var, None)
            st.session_state["ivs"].pop(var, None)
            st.success(f"Recategorização de {var} resetada!")

        if st.session_state["resultados"]:
            st.subheader("📌 Variáveis já categorizadas")
            df_status = pd.DataFrame({
                "Variável": list(st.session_state["ivs"].keys()),
                "IV_total": list(st.session_state["ivs"].values())
            }).sort_values("IV_total", ascending=False).reset_index(drop=True)
            st.dataframe(df_status)

        if st.sidebar.button("🔄 Resetar todas"):
            st.session_state["resultados"] = {}
            st.session_state["ivs"] = {}
            st.success("Todas as categorizações foram resetadas!")

        if st.sidebar.button("⬇️ Exportar base final"):
            df_export = df.copy()
            for var, categorias in st.session_state["resultados"].items():
                df_export[f"{var}_cat"] = categorias.values
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download base_final.csv", csv, "base_final.csv", "text/csv")
