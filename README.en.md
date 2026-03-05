# sp-fitting-models

English README for the project.

A Python library for fitting supramolecular polymerization data with various thermodynamic models.

## Overview

This library provides mathematical models for analyzing supramolecular polymerization data. It enables quantitative analysis of temperature-dependent aggregation behavior and estimation of thermodynamic parameters (enthalpy and entropy).

## Features

- **Multiple model families**
  - **Isodesmic model**: a simple association model where all association steps share the same equilibrium constant
  - **Cooperative model**: a cooperative association model with different constants for nucleation and elongation
  - **Mixed model**: a competing two-pathway model combining isodesmic and cooperative mechanisms

- **Temperature-dependent analysis**
  - Temperature-dependent equilibrium constants based on the van’t Hoff relationship
  - Estimation of $\Delta H$ (enthalpy change) and $\Delta S$ (entropy change)

- **Fitting utilities**
  - Experimental-data fitting using `lmfit`
  - Simultaneous fitting across multiple concentrations (global fitting)

## Installation

```bash
uv add https://github.com/IndigoCarmine/sp_fitting_models.git
```

## Usage

### Basic example

```python
import numpy as np
import matplotlib.pyplot as plt
from sp_fitting_models.models import temp_cooperative_model

# Temperature range
temps = np.linspace(280, 400, 200)  # 280-400 K

# Thermodynamic parameters
deltaH = -96000      # Enthalpy change (J/mol)
deltaS = -180        # Entropy change (J/(mol·K))
deltaHnuc = 100000   # Nucleation penalty (J/mol)
c_tot = 5e-6         # Total concentration (M)

# Calculate aggregation
agg = temp_cooperative_model(
    Temp=temps,
    deltaH=deltaH,
    deltaS=deltaS,
    deltaHnuc=deltaHnuc,
    c_tot=c_tot,
    scaler=1.0
)

# Plot
plt.plot(temps - 273.15, agg)
plt.xlabel('Temperature (°C)')
plt.ylabel('Aggregation')
plt.show()
```

### Data fitting

```python
import lmfit as lm
from sp_fitting_models.data import TempVsAggData
from sp_fitting_models.fitting import objective_temp_cooperative

# Prepare your experimental data
data_list = [
    TempVsAggData(temp=temps1, agg=agg1, concentration=c1),
    TempVsAggData(temp=temps2, agg=agg2, concentration=c2),
]

# Set up parameters
params = lm.Parameters()
params.add('deltaH', value=-100000, min=-200000, max=0)
params.add('deltaS', value=-180, min=-400, max=0)
params.add('deltaHnuc', value=50000, min=0, max=200000)
params.add('scaler', value=1.0, min=0.5, max=1.5)

# Fit
minner = lm.Minimizer(objective_temp_cooperative, params, fcn_args=(data_list,))
result = minner.minimize()

print(lm.fit_report(result))
```

### Interactive visualization

```bash
# Run the interactive mixed-model example
python examples/interactive_mixed.py
```

Use sliders to change parameters and observe aggregation-curve changes in real time.

### Build as a Windows app (uv + PyInstaller)

You can build `examples/interactive_mixed.py` as a windowed Windows GUI app (no console).

```powershell
./scripts/build_interactive_mixed.ps1
```

Or from `cmd.exe`:

```bat
build_interactive_mixed.bat
```

Output:

- `dist/interactive_mixed/interactive_mixed.exe`

This script performs:

1. `uv sync` to sync dependencies and the local package
2. `uv run --with pyinstaller ...` to build the GUI app

## Project structure

```
sp_fitting_models/
├── src/
│   └── sp_fitting_models/
│       ├── __init__.py
│       ├── data.py              # Data structures
│       ├── models/              # Model implementations
│       │   ├── __init__.py
│       │   ├── isodesmic.py     # Isodesmic models
│       │   ├── cooperative.py   # Cooperative models
│       │   ├── mixed.py         # Mixed models
│       │   └── utils.py         # Utility functions
│       └── fitting/             # Fitting utilities
│           ├── __init__.py
│           └── objective.py     # Objective functions for lmfit
├── tests/                       # Test files
│   ├── test_isodesmic.py
│   ├── test_cooperative.py
│   ├── test_mixed.py
│   └── test_fitting.py
├── examples/                    # Example scripts
│   ├── basic_usage.py
│   └── interactive_mixed.py
├── pyproject.toml
└── README.md
```

## Model description

### Isodesmic model

A model in which all association steps share the same equilibrium constant $K$. It typically yields a sigmoidal aggregation curve.

$$K = \exp\left(-\frac{\Delta H}{RT} + \frac{\Delta S}{R}\right)$$

The equilibrium scheme is:

$$M \stackrel{K}{\rightleftarrows}
 M_2\stackrel{K}{\rightleftarrows}
 M_3 \stackrel{K}{\rightleftarrows} ...$$

### Cooperative model

A model with distinct equilibrium constants for nucleation and elongation. Cooperativity is represented by a nucleation penalty $\sigma$.

$$\sigma = \exp\left(-\frac{\Delta H_{nuc}}{RT}\right)$$

$$K = \exp\left(-\frac{\Delta H}{RT} + \frac{\Delta S}{R}\right)$$

$$K_{nuc} = \sigma K$$

The equilibrium scheme is:

$$M \stackrel{K_{nuc}}{\rightleftarrows}
 M_2\stackrel{K}{\rightleftarrows}
 M_3 \stackrel{K}{\rightleftarrows} ...$$

### Mixed model

A two-pathway model in which isodesmic and cooperative routes compete while sharing the same monomer pool.

The model considers equilibria such as:

$$M \stackrel{K_{nuc}}{\rightleftarrows}
 M_2\stackrel{K}{\rightleftarrows}
 M_3 \stackrel{K}{\rightleftarrows} ...$$
$$ \searrow \nwarrow^{K_{iso}}
 M_2\stackrel{K_{iso}}{\rightleftarrows}
  M_3 \stackrel{K_{iso}}{\rightleftarrows} ...$$

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run a specific test
python tests/test_cooperative.py
```

## Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Interactive mixed-model visualization
python examples/interactive_mixed.py
```

## Dependencies

- Python >= 3.13
- numpy >= 2.4.2
- numba >= 0.64.0
- lmfit >= 1.3.4
- matplotlib >= 3.10.8

## Citation

I would be grateful if you cite this library in your publications, but it is not mandatory. Please feel free to use it as you see fit.

## Author
Yuhei Yamada (Orcid: [0009-0003-9780-4135](https://orcid.org/0009-0003-9780-4135), google scholar: [Yuhei Yamada](https://scholar.google.co.jp/citations?user=mRKL6CYAAAAJ))