# Copyright (c) 2025 takotime808

from jinja2 import Template

template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ project_title or "Multi-Fidelity, Multi-Output Surrogate Modeling Report" }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        h1, h2, h3 { color: #2d4059; }
        .section { margin-bottom: 2em; }
        .plot { max-width: 800px; margin: 1em 0; }
        table { border-collapse: collapse; margin: 1em 0; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background: #f4f4f4; }
    </style>
</head>
<body>
    <h1>{{ project_title or "Multi-Fidelity, Multi-Output Surrogate Modeling Report" }}</h1>
    <div class="section">
        <h2>Overview</h2>
        <p>
            <b>Model type:</b> {{ model_type }}<br>
            <b>Fidelity levels:</b> {{ fidelity_levels|join(", ") if fidelity_levels else "N/A" }}<br>
            <b>Outputs:</b> {{ output_names|join(", ") if output_names else "N/A" }}<br>
            <b>Description:</b> {{ description or "No description provided." }}
        </p>
    </div>
    <div class="section">
        <h2>Regression Metrics</h2>
        <table>
            <tr>
                <th>Output</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Other</th>
            </tr>
            {% for name in output_names %}
            <tr>
                <td>{{ name }}</td>
                <td>{{ metrics[name]['r2']|round(3) }}</td>
                <td>{{ metrics[name]['rmse']|round(3) }}</td>
                <td>{{ metrics[name]['mae']|round(3) }}</td>
                <td>
                    {% for k,v in metrics[name].items() %}
                        {% if k not in ['r2','rmse','mae'] %}
                            <b>{{ k }}:</b> {{ v|round(3) }}<br>
                        {% endif %}
                    {% endfor %}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    <div class="section">
        <h2>Uncertainty Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {% for k, v in uncertainty_metrics.items() %}
            <tr>
                <td>{{ k }}</td>
                <td>{{ v|round(3) }}</td>
            </tr>
            {% endfor %}
        </table>
        <h3>Uncertainty Toolbox Plots</h3>
        {% for p in uncertainty_plots %}
        <div class="plot">
            <img src="data:image/png;base64,{{ p.img_b64 }}" alt="{{ p.title }}" style="width: 100%; max-width: 800px;">
            <div><b>{{ p.title }}</b></div>
            <div>{{ p.caption }}</div>
        </div>
        {% endfor %}
    </div>
    <div class="section">
        <h2>Predictions vs. True Values (per output)</h2>
        {% for name in output_names %}
        <h3>{{ name }}</h3>
        <div class="plot">
            <img src="data:image/png;base64,{{ prediction_plots[name] }}" alt="Pred vs True for {{ name }}">
        </div>
        {% endfor %}
    </div>
    <div class="section">
        <h2>SHAP Explanations</h2>
        {% for name, img in shap_plots.items() %}
        <h3>{{ name }}</h3>
        <div class="plot">
            <img src="data:image/png;base64,{{ img }}" alt="SHAP for {{ name }}">
        </div>
        {% endfor %}
    </div>
    <div class="section">
        <h2>Partial Dependence Plots (PDP)</h2>
        {% for name in output_names %}
        <h3>{{ name }}</h3>
        <div class="plot">
            <img src="data:image/png;base64,{{ pdp_plots[name] }}" alt="PDP for {{ name }}">
        </div>
        {% endfor %}
    </div>
    {% if pca_explained_variance %}
    <div class="section">
        <h2>PCA Explained Variance</h2>
        {% if pca_n_components %}
        <p>
            <i>
                Components retained: {{ pca_n_components }}{% if pca_method %} (method: {{ pca_method }}{% if pca_threshold %}, threshold {{ pca_threshold|round(2) }}{% endif %}){% endif %}.
            </i>
        </p>
        {% endif %}
        <div class="plot">
            {% if pca_variance_plot %}
            <img src="data:image/png;base64,{{ pca_variance_plot }}" alt="PCA Variance">
            {% endif %}
        </div>
        <table>
            <tr><th>Component</th><th>Explained Variance Ratio</th></tr>
            {% for var in pca_explained_variance %}
            <tr><td>PC{{ loop.index }}</td><td>{{ var|round(3) }}</td></tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    <div class="section">
        <h2>Sampling & Input Space Visualization</h2>
        <div>
            <h3>Sampling Technique Inference</h3>
            <div class="plot">
                <img src="data:image/png;base64,{{ sampling_umap_plot }}" alt="UMAP Sampling Plot">
            </div>
            <div>{{ sampling_method_explanation }}</div>
        </div>
        {% if sampling_other_plots %}
        {% for p in sampling_other_plots %}
        <div class="plot">
            <img src="data:image/png;base64,{{ p.img_b64 }}" alt="{{ p.title }}">
            <div><b>{{ p.title }}</b></div>
        </div>
        {% endfor %}
        {% endif %}
    </div>
    <div class="section">
        <h2>Additional Diagnostics</h2>
        {% for p in other_plots %}
        <div class="plot">
            <img src="data:image/png;base64,{{ p.img_b64 }}" alt="{{ p.title }}">
            <div><b>{{ p.title }}</b></div>
            <div>{{ p.caption }}</div>
        </div>
        {% endfor %}
    </div>
    <div class="section">
        <h2>Appendix: Methodology & Notes</h2>
        <p>
            <b>Training data size:</b> {{ n_train }}<br>
            <b>Test data size:</b> {{ n_test }}<br>
            <b>Cross-validation:</b> {{ cross_validation }}<br>
            <b>Random seed:</b> {{ seed }}
        </p>
        <p>
            <b>Notes:</b><br>
            {{ notes|replace('\n','<br>') }}
        </p>
    </div>
</body>
</html>
""")