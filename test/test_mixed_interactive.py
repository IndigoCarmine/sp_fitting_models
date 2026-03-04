import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sp_fitting_models.mixed_models import temp_coop_iso_model
import sp_fitting_models.data as data
from sp_fitting_models.models import temp_cooperative_model, temp_isodesmic_model

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

temps = np.linspace(280, 400, 200)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.45)


# Initial plot
def plot_curve(params):
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
    ax.clear()
    ax.plot(temps - 273.15, agg, label=f"Conc={params['concentration']:.2e} M")

    # isodesmic only
    agg_iso = temp_isodesmic_model(
        Temp=temps,
        deltaH=params["iso_deltaH"],
        deltaS=params["iso_deltaS"],
        c_tot=params["concentration"],
    )
    ax.plot(temps - 273.15, agg_iso, label=f"Conc={params['concentration']:.2e} M (isodesmic)", linestyle="--")

    # # cooperative only
    # agg_coop = temp_cooperative_model(
    #     Temp=temps,
    #     deltaH=params["co_deltaH"],
    #     deltaS=params["co_deltaS"],
    #     deltaHnuc=params["co_deltaHnuc"],
    #     c_tot=params["concentration"],
    # )
    # ax.plot(temps - 273.15, agg_coop, label=f"Conc={params['concentration']:.2e} M (cooperative)", linestyle=":")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.legend()
    fig.canvas.draw_idle()


plot_curve(defaults)

# Sliders
axcolor = "lightgoldenrodyellow"
ax_co_deltaH = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_co_deltaS = plt.axes([0.25, 0.31, 0.65, 0.03], facecolor=axcolor)
ax_co_deltaHnuc = plt.axes([0.25, 0.27, 0.65, 0.03], facecolor=axcolor)
ax_iso_deltaH = plt.axes([0.25, 0.19, 0.65, 0.03], facecolor=axcolor)
ax_iso_deltaS = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_concentration = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)

s_co_deltaH = Slider(ax_co_deltaH, "coop_deltaH", -120000, 120000, valinit=defaults["co_deltaH"])
s_co_deltaS = Slider(ax_co_deltaS, "coop_deltaS", -300, 300, valinit=defaults["co_deltaS"])
s_co_deltaHnuc = Slider(ax_co_deltaHnuc, "coop_deltaHnuc", 0, 150000, valinit=defaults["co_deltaHnuc"])
s_iso_deltaH = Slider(ax_iso_deltaH, "iso_deltaH", -120000, 120000, valinit=defaults["iso_deltaH"])
s_iso_deltaS = Slider(ax_iso_deltaS, "iso_deltaS", -300, 300, valinit=defaults["iso_deltaS"])
s_concentration = Slider(
    ax_concentration, "concentration", 1e-7, 1e-5, valinit=defaults["concentration"], valfmt="%1.2e"
)


def update(val):
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


s_co_deltaH.on_changed(update)
s_co_deltaS.on_changed(update)
s_co_deltaHnuc.on_changed(update)
s_iso_deltaH.on_changed(update)
s_iso_deltaS.on_changed(update)
s_concentration.on_changed(update)

plt.show()
