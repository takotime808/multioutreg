# Copyright (c) 2025 takotime808

# metrics_df, overall_metrics = get_uq_performance_metrics_flexible(multi_gp, X_test, Y_test)

# print("Available columns:", metrics_df.columns)
# metrics_to_plot = [m for m in ['rmse', 'mae', 'nll', 'miscal_area'] if m in metrics_df.columns]
# if not metrics_to_plot:
#     print("No matching metrics found in metrics_df. Available columns:", metrics_df.columns)
# else:
#     ax = metrics_df[metrics_to_plot].plot.bar(figsize=(10, 6))
#     ax.set_xticklabels([f"Output {i}" for i in metrics_df['output']])
#     plt.xlabel('Output')
#     plt.title('Uncertainty Toolbox Metrics per Output')
#     plt.legend(title="Metric")
#     plt.tight_layout()
#     plt.show()
