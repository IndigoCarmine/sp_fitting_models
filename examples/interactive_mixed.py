"""
Interactive visualization of the mixed cooperative-isodesmic model.

This script provides an interactive plot with sliders to explore how different
thermodynamic parameters affect the aggregation behavior of the mixed model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sp_fitting_models.models import (
    temp_coop_iso_model,
    temp_cooperative_model,
    temp_isodesmic_model,
)


# Initial parameter values
defaults = {
    "co_deltaH": -96000,
    "co_deltaS": -180,
    "co_deltaHnuc": 100000,
    "co_scaler": 1.0,
    "iso_deltaH": -96000,
    "iso_deltaS": -195,
    "concentration": 1e-6,
}

temps = np.linspace(280, 500, 200)

fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.25, bottom=0.45)


# Plot function
def plot_curve(params):
    """Update the plot with new parameters."""
    # Mixed model
    agg = temp_coop_iso_model(
        Temp=temps,
        deltaH_iso=params["iso_deltaH"],
        deltaS_iso=params["iso_deltaS"],
        deltaH_coop=params["co_deltaH"],
        deltaS_coop=params["co_deltaS"],
        deltaHnuc_coop=params["co_deltaHnuc"],
        c_tot=params["concentration"],
        scaler=params["co_scaler"],
    )

    # Isodesmic only (for comparison)
    agg_iso = temp_isodesmic_model(
        Temp=temps,
        deltaH=params["iso_deltaH"],
        deltaS=params["iso_deltaS"],
        c_tot=params["concentration"],
    )

    ax.clear()
    ax.plot(temps - 273.15, agg, label="Mixed Model", linewidth=2, color="blue")
    ax.plot(
        temps - 273.15,
        agg_iso,
        label="Isodesmic Only",
        linestyle="--",
        linewidth=2,
        color="orange",
    )

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Aggregation", fontsize=12)
    ax.set_title(f"Mixed Model (Conc={params['concentration']:.2e} M)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    fig.canvas.draw_idle()


# Initial plot
plot_curve(defaults)

# Create sliders
axcolor = "lightgoldenrodyellow"
ax_co_deltaH = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_co_deltaS = plt.axes([0.25, 0.31, 0.65, 0.03], facecolor=axcolor)
ax_co_deltaHnuc = plt.axes([0.25, 0.27, 0.65, 0.03], facecolor=axcolor)
ax_iso_deltaH = plt.axes([0.25, 0.19, 0.65, 0.03], facecolor=axcolor)
ax_iso_deltaS = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_concentration = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)

s_co_deltaH = Slider(ax_co_deltaH, "Coop ΔH", -120000, 120000, valinit=defaults["co_deltaH"])
s_co_deltaS = Slider(ax_co_deltaS, "Coop ΔS", -300, 300, valinit=defaults["co_deltaS"])
s_co_deltaHnuc = Slider(ax_co_deltaHnuc, "Coop ΔHnuc", 0, 150000, valinit=defaults["co_deltaHnuc"])
s_iso_deltaH = Slider(ax_iso_deltaH, "Iso ΔH", -120000, 120000, valinit=defaults["iso_deltaH"])
s_iso_deltaS = Slider(ax_iso_deltaS, "Iso ΔS", -300, 300, valinit=defaults["iso_deltaS"])
s_concentration = Slider(
    ax_concentration,
    "Concentration",
    1e-7,
    1e-5,
    valinit=defaults["concentration"],
    valfmt="%1.2e",
)


# Update function
def update(val):
    """Update the plot when a slider is changed."""
    params = {
        "co_deltaH": s_co_deltaH.val,
        "co_deltaS": s_co_deltaS.val,
        "co_deltaHnuc": s_co_deltaHnuc.val,
        "co_scaler": 1.0,
        "iso_deltaH": s_iso_deltaH.val,
        "iso_deltaS": s_iso_deltaS.val,
        "concentration": s_concentration.val,
    }
    plot_curve(params)


# Connect sliders to update function
s_co_deltaH.on_changed(update)
s_co_deltaS.on_changed(update)
s_co_deltaHnuc.on_changed(update)
s_iso_deltaH.on_changed(update)
s_iso_deltaS.on_changed(update)
s_concentration.on_changed(update)

plt.show()
