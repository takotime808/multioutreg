# Copyright (c) 2025 takotime808

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File upload UI element
uploaded_file = st.file_uploader("Upload your regression dataset (CSV)", type='csv')

def plot_pairgrid_with_kde(df, num_cols):
    st.subheader("Pairwise Relationships (KDE on Lower Triangle)")

    # Only plot if we have at least two numeric columns
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for pairwise visualization.")
        return

    # Create the PairGrid with only the numerical columns
    g = sns.PairGrid(df[num_cols])

    # Scatter in the upper, KDE in the lower, histogram in the diagonal
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot, cmap="Blues", fill=True)
    g.map_diag(sns.histplot, edgecolor="black")

    plt.tight_layout()
    return g

def infer_sampling(df, num_cols):
    # Grid: unique value count per feature matches grid size, mesh pattern in plots
    n_unique_vals = [len(df[col].unique()) for col in num_cols]
    grid_like = np.prod(n_unique_vals) == len(df)
    # LHS: unique values per feature, each only used once; uniform histogram
    lhs_like = all(df[col].value_counts().max() == 1 for col in num_cols)
    # For 'other', neither pattern
    if grid_like:
        st.info("Sampling looks like grid sampling (regular mesh in value pairs, distinct fixed values per feature).")
    elif lhs_like:
        st.info("Sampling looks like Latin hypercube sampling (one value per bin per feature, uniform marginals).")
    else:
        st.info("Sampling does not match grid or LHS. It may be a random or other custom scheme.")

def main():
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for sampling pattern analysis.")
        else:
            g = plot_pairgrid_with_kde(df, num_cols)
            st.pyplot(g.fig)
            infer_sampling(df, num_cols)
    else:
        st.info("Please upload a CSV file with your regression data.")

if __name__ == "__main__":
    main()