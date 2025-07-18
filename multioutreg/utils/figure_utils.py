# Copyright (c) 2025 takotime808

import base64
import io
import matplotlib.pyplot as plt
from typing import Callable, Any

def safe_plot_b64(
    plot_func: Callable[..., Any], 
    *args: Any, 
    **kwargs: Any
) -> str:
    """
    Executes a plotting function and returns a PNG plot as a base64-encoded string.
    
    Args:
        plot_func: A function that generates a matplotlib plot (e.g., a plotting routine).
        *args: Positional arguments to pass to `plot_func`.
        **kwargs: Keyword arguments to pass to `plot_func`.
    
    Returns:
        A base64-encoded PNG image of the generated plot. If the plotting function
        raises an exception, returns a base64-encoded blank plot with the error message.
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
