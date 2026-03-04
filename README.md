# sp-fitting-models

超分子ポリマーのフィッティングモデルライブラリ

A Python library for fitting supramolecular polymerization data with various thermodynamic models.

## 概要 / Overview

このライブラリは、超分子ポリマー形成データを解析するための数理モデルを提供します。特に温度依存的な会合挙動を定量的に解析し、熱力学パラメータ（エンタルピー、エントロピー）を推定することができます。

This library provides mathematical models for analyzing supramolecular polymerization data. It enables quantitative analysis of temperature-dependent aggregation behavior and estimation of thermodynamic parameters (enthalpy, entropy).

## 特徴 / Features

- **複数のモデルに対応**
  - **Isodesmicモデル**: すべての会合定数が等しい単純な会合モデル
  - **Cooperativeモデル**: 核形成と伸長で異なる定数を持つ協同的会合モデル
  - **Mixedモデル**: IsodesmicとCooperativeの2経路が競合するモデル

- **温度依存性の解析**
  - van't Hoff式に基づく温度依存的な会合定数の計算
  - ΔH（エンタルピー変化）とΔS（エントロピー変化）の推定

- **フィッティング機能**
  - lmfitライブラリを使用した実験データへのフィッティング
  - 複数濃度データの同時フィッティング(グローバルフィット)に対応


## インストール / Installation

```bash
uv add https://github.com/IndigoCarmine/sp_fitting_models.git
```

## 使用方法 / Usage

### 基本的な使用例

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

### データフィッティング

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

### インタラクティブな可視化

```bash
# Run the interactive mixed model example
python examples/interactive_mixed.py
```

スライダーを使用してパラメータを変更し、リアルタイムで会合曲線の変化を観察できます。

## プロジェクト構造 / Project Structure

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

## モデルの説明 / Model Description

### Isodesmicモデル

すべての会合ステップが同じ平衡定数Kを持つモデルです。シグモイド型の会合曲線を示します。

$$K = \exp\left(-\frac{\Delta H}{RT} + \frac{\Delta S}{R}\right)$$

次のような平衡状態です。
供給されるモノマーは省略してあります。：
$$M \stackrel{K}{\rightleftarrows}
 M_2\stackrel{K}{\rightleftarrows}
 M_3 \stackrel{K}{\rightleftarrows} ...$$

### Cooperativeモデル

核形成と伸長で異なる平衡定数を持つモデルです。非シグモイド型の会合曲線を示します。核形成ペナルティσにより協同性が表現されます。

$$\sigma = \exp\left(-\frac{\Delta H_{nuc}}{RT}\right)$$

$$ K = \exp\left(-\frac{\Delta H}{RT} + \frac{\Delta S}{R}\right)$$

$$ K_{nuc} = \sigma K $$
次のような平衡状態です。
$$M \stackrel{K_{nuc}}{\rightleftarrows}
 M_2\stackrel{K}{\rightleftarrows}
 M_3 \stackrel{K}{\rightleftarrows} ...$$


### Mixedモデル

IsodesmicとCooperativeの2つの経路が同じモノマープールを共有して競合するモデルです。実験系で複数の会合機構が同時に起こる場合に適用できます。

次のような平衡状態を考えています。
$$M \stackrel{K_{nuc}}{\rightleftarrows}
 M_2\stackrel{K}{\rightleftarrows}
 M_3 \stackrel{K}{\rightleftarrows} ...$$
$$ \searrow \nwarrow^{K_{iso}}
 M_2\stackrel{K_{iso}}{\rightleftarrows}
  M_3 \stackrel{K_{iso}}{\rightleftarrows} ...$$
（MDではこれ以上きれいに書けませんでした...）
## テスト / Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_cooperative.py
```

## サンプル / Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Interactive mixed model visualization
python examples/interactive_mixed.py
```

## 依存関係 / Dependencies

- Python >= 3.13
- numpy >= 2.4.2
- numba >= 0.64.0
- lmfit >= 1.3.4
- matplotlib >= 3.10.8

## 引用　/Citation

書いていただけるなら嬉しいですが、必ずしも論文で言及する必要はありません。
ご自由にお使いください。

I would be grateful if you could cite this library in your publications, but it is not mandatory. Please feel free to use it as you see fit.