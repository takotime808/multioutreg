# Copyright (c) 2025 takotime808

import base64
import io
import matplotlib.pyplot as plt

def safe_plot_b64(plot_func, *args, **kwargs):
    """
    Calls plot_func and returns a base64 string of the plot.
    If plotting fails, returns a blank plot with an error message.
    """
    try:
        plot_func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        # Return a blank plot with error message
        fig, ax = plt.subplots(figsize=(6,2))
        ax.axis('off')
        ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', color='red', wrap=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
