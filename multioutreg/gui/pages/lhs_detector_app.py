# Copyright (c) 2025 takotime808
import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from multioutreg.sampling.detect import detect_sampling_method, guess_input_columns

# --------- Utility Functions ----------

def plot_pairwise(df):
    fig = sns.pairplot(df)
    st.pyplot(fig)

def plot_histograms(df):
    fig, axs = plt.subplots(nrows=len(df.columns), figsize=(8, 2 * len(df.columns)), squeeze=False)
    for i, col in enumerate(df.columns):
        axs[i, 0].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axs[i, 0].set_title(f"Histogram of {col}")
    plt.tight_layout()
    st.pyplot(fig)

def plot_correlation_matrix(df):
    corr = df.corr(method='spearman')
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    return corr

# --------- Streamlit UI ----------

st.title("🧪 LHS Detection Tool")
st.write("""
This app helps you evaluate whether uploaded data might have been generated using **Latin Hypercube Sampling (LHS)**.
It combines visual inspection, correlation analysis, and uniformity metrics.
""")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header("🔍 Data Summary and Preprocessing")
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_cols:
        st.warning(f"⚠️ Categorical columns excluded: {', '.join(categorical_cols)}")

    default_inputs = guess_input_columns(numeric_cols)
    input_cols = st.multiselect(
        "Input (design) columns",
        numeric_cols,
        default=default_inputs,
        help="Only the sampled inputs form the design. Response columns are not sampled and "
             "make any design look non-LHS, so leave them out.",
    )
    numeric_df = df[input_cols].dropna().copy()

    constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() == 1]
    if constant_cols:
        st.warning(f"⚠️ Constant columns dropped: {', '.join(constant_cols)}")
        numeric_df = numeric_df.drop(columns=constant_cols)

    if numeric_df.empty or len(numeric_df) < 4:
        st.error("❌ Need at least one non-constant input column and 4 complete rows.")
        st.stop()

    # Normalize to [0, 1]
    scaled_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    scaled_df = scaled_df.fillna(0.5)  # fallback in case of zero division
    st.success("✅ Numeric data normalized to [0, 1] for LHS analysis.")

    # --- Visual Inspection ---
    st.subheader("1. 📊 Visual Inspection")
    st.write("### Pairwise Relationships")
    plot_pairwise(scaled_df)

    st.write("### Marginal Distributions")
    plot_histograms(scaled_df)

    # --- Correlation Analysis ---
    st.subheader("2. 🔗 Correlation Matrix (Spearman)")
    corr = plot_correlation_matrix(scaled_df)
    st.write("Inputs of a designed experiment (LHS, Sobol, random or grid) should be nearly uncorrelated.")

    # --- Uniformity Metrics ---
    st.subheader("3. 📏 Uniformity Metrics")
    with st.spinner("Comparing with simulated Random / LHS / Sobol designs..."):
        result = detect_sampling_method(numeric_df.values, random_state=0)

    if result.p_values:
        st.markdown(f"""
        Your design was compared with **{result.n_sim}** designs of the same size simulated by each
        method. A p-value is the share of those designs that look at least as unusual as yours;
        a small p-value (< 0.05) rules that method out.
        """)
        st.table(pd.DataFrame(
            {"p-value": result.p_values}
        ).rename_axis("Method").sort_values("p-value", ascending=False))
        stats = result.statistics
        st.markdown(
            f"- **Strata filled** (mean over inputs): `{stats['strata']:.3f}` "
            "(LHS fills nearly all of them after rescaling; random fills about 0.63)\n"
            + (f"- **Centered L2 discrepancy** (lower = more uniform): `{10 ** stats['log_cd']:.3e}`\n"
               if "log_cd" in stats else "")
            + f"- **Mean nearest-neighbour distance**: `{stats['nn']:.4f}`"
        )

    # --- Interpretation Help ---
    st.subheader("4. 🧠 Interpretation Guide")
    st.markdown("""
    - **Histograms** of an LHS (or Sobol) design are flat; every one of the n equal-width strata
      of each input holds exactly one point.
    - **Correlation** between inputs should be near zero for every method; it does not separate
      LHS from random.
    - The test assumes **uniform inputs on a box**. Designs with other marginals (e.g. an LHS of
      normal variables) look like none of the methods here; use
      `multioutreg compare_sample_methods` with a YAML of the distributions instead.
    - With few points several methods can be consistent with the data at once.
    """)

    # --- Final Determination ---
    st.subheader("5. 🏁 Final Determination")
    mean_corr = np.mean(np.abs(corr.values[np.triu_indices_from(corr, 1)])) if corr.shape[0] > 1 else 0.0
    st.write(f"**Mean absolute correlation:** `{mean_corr:.4f}`")
    st.caption(result.explanation)

    if result.method == "LHS":
        st.success("✅ **The data is consistent with LHS.**")
    elif result.method == "Grid":
        st.warning("⚠️ **The data appears to be generated from a structured grid.**")
    elif result.method == "Sobol":
        st.warning("⚠️ **The data looks like a low-discrepancy (Sobol / Halton) sequence, not LHS.**")
    elif result.method == "Random":
        st.error("❌ **The data is consistent with random (i.i.d. uniform) sampling, not LHS.**")
    else:
        st.error("❌ **The data does not match LHS, a grid, Sobol or random uniform sampling.**")

else:
    st.info("⬆️ Upload a CSV file to begin.")
