# Copyright (c) 2025 takotime808

import streamlit as st
import pandas as pd
import tempfile
from typer.testing import CliRunner
from mor_cli.detect_sampling import app as detect_app

st.title("Sampling Method Detection")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        runner = CliRunner()
        result = runner.invoke(detect_app, [tmp.name])
        if result.exit_code == 0:
            st.text(result.stdout)
        else:
            st.error(result.stdout or result.stderr)