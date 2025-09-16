import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="App para Categorização", layout="wide")

# ==========================================
# Função auxiliar - binning inicial
# ==========================================


def binarizar_var(df, var, target, q=5):
    df_aux = df.copy()
    df_aux["faixa"] = pd.qcut(df_aux[var].dropna(), q=q, duplicates="drop")
    df_aux["faixa"] = df_aux["faixa"].astype(str)
    df_aux.loc[df_aux[var].isna(), "faixa"] = "-99"
    return df_aux

# ==========================================
# Função auxiliar - tabela consolidada
# ==========================================


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

    # WOE e IV
    def calc_woe(p_b, p_m):
        if p_b > 0 and p_m > 0:
            return math.log(p_b / p_m)
        else:
            return 0

    tab["WOE"] = tab.apply(lambda r: calc_woe(
        r["P_bons"], r["P_maus"]), axis=1)
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

    tab = pd.concat([tab, total_row], ignore_index=True)
    return tab

# ==========================================
# Função auxiliar - gráfico por safra
# ==========================================


def plot_taxa_por_safra(df, var_cat, target, safra_col="safra"):
    resumo = (
        df.groupby([safra_col, var_cat])[target]
        .mean()
        .reset_index()
    )
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
# Streamlit App
# ==========================================
st.title("📊 App para Categorização")

file = st.file_uploader("Carregue sua base (CSV)", type="csv")

# Inicializar session_state
if "resultados" not in st.session_state:
    st.session_state["resultados"] = {}
if "ivs" not in st.session_state:
    st.session_state["ivs"] = {}

if file is not None:
    df = pd.read_csv(file)
    st.write("✅ Base carregada:", df.shape)

    # Seleção dinâmica do target e da safra
    st.subheader("⚙️ Configurações principais")
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Selecione a coluna de target", df.columns)
    with col2:
        safra_col = st.selectbox("Selecione a coluna de safra (ou deixe em branco)", [
                                 ""] + df.columns.tolist())

    # 🔎 Validação do target: precisa ser binário
    valores_unicos = df[target].dropna().unique()
    if len(valores_unicos) > 2:
        st.error(
            f"❌ A coluna '{target}' tem mais de 2 valores únicos ({valores_unicos}). Selecione uma coluna binária (0/1).")
        st.stop()

    # Se safra não for escolhida -> escolher coluna de data e criar safra YYYY-MM
    data_col = None
    if safra_col == "":
        st.info(
            "Nenhuma coluna de safra selecionada. Escolha uma coluna de data para gerar a safra (YYYY-MM).")
        data_col = st.selectbox("Selecione a coluna de data", df.columns)

        try:
            df[data_col] = pd.to_datetime(df[data_col], errors="coerce")
            df["safra"] = df[data_col].dt.to_period("M").astype(str)
            safra_col = "safra"
            st.success(
                f"✅ Coluna de safra criada a partir de '{data_col}' no formato YYYY-MM.")
        except Exception as e:
            st.error(f"Erro ao converter coluna {data_col} em datas: {e}")
            st.stop()

    # ==========================
    # Escolha dinâmica das colunas a descartar
    # ==========================
    st.subheader("⚙️ Colunas para descartar")

    # Remover target, safra e data_col da lista
    excluidas = [target, safra_col]
    if data_col:
        excluidas.append(data_col)

    todas_cols = [c for c in df.columns if c not in excluidas]

    cols_drop = st.multiselect(
        "Selecione as colunas que NÃO serão categorizadas",
        todas_cols,
        default=[]
    )

    # Lista de variáveis disponíveis
    variaveis_analise = [c for c in todas_cols if c not in cols_drop]

    var = st.selectbox("Selecione a variável para análise", variaveis_analise)

    if var:
        df_aux = binarizar_var(df, var, target)

        # Tabela inicial
        st.subheader("📊 Tabela consolidada (auto-binning inicial)")
        tab_ini = tabela_consolidada(df_aux, "faixa", target)
        st.dataframe(tab_ini)

        # Reagrupamento manual
        st.subheader("✏️ Reagrupamento manual")
        categorias = tab_ini.loc[tab_ini["faixa"] != "TOTAL", "faixa"].tolist()
        grupos = {}
        for cat in categorias:
            novo_grupo = st.text_input(
                f"Defina grupo para {cat} ({var})", value=cat, key=f"{var}_{cat}")
            grupos[cat] = novo_grupo

        df_aux["faixa_final"] = df_aux["faixa"].map(grupos)

        # Tabela final
        st.subheader("📊 Tabela consolidada (após reagrupamento)")
        tab_final = tabela_consolidada(df_aux, "faixa_final", target)
        st.dataframe(tab_final)

        # Gráfico
        st.subheader("📈 Taxa de default por safra")
        plot_taxa_por_safra(df_aux, "faixa_final", target, safra_col)

        # Salvar no repositório interno
        if st.button(f"💾 Salvar recategorização de {var}"):
            st.session_state["resultados"][var] = df_aux["faixa_final"].copy()
            st.session_state["ivs"][var] = tab_final.loc[tab_final["faixa_final"]
                                                         == "TOTAL", "IV"].values[0]
            st.success(f"Recategorização de {var} salva!")

        # Resetar apenas a variável atual
        if st.button(f"♻️ Resetar {var}"):
            if var in st.session_state["resultados"]:
                del st.session_state["resultados"][var]
            if var in st.session_state["ivs"]:
                del st.session_state["ivs"][var]
            st.success(f"Recategorização de {var} foi resetada!")

    # 📌 Listar variáveis já categorizadas com IV total
    if st.session_state["resultados"]:
        st.subheader("📌 Variáveis já categorizadas")
        df_status = pd.DataFrame({
            "Variável": list(st.session_state["ivs"].keys()),
            "IV_total": list(st.session_state["ivs"].values())
        }).sort_values("IV_total", ascending=False).reset_index(drop=True)
        st.dataframe(df_status)

    # Resetar todas
    if st.button("🔄 Resetar todas as categorizações"):
        st.session_state["resultados"] = {}
        st.session_state["ivs"] = {}
        st.success("Todas as categorizações foram resetadas!")

    # Exportar base final consolidada
    if st.button("⬇️ Exportar base final consolidada"):
        df_export = df.copy()
        for var, categorias in st.session_state["resultados"].items():
            df_export[f"{var}_cat"] = categorias.values
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download base_final.csv",
            data=csv,
            file_name="base_final.csv",
            mime="text/csv",
        )
