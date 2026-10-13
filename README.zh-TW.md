# FuncPie

正體中文 | [English](README.md)

[![CI](https://github.com/ai-tech-tw/func_pie/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-tech-tw/func_pie/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked](https://img.shields.io/badge/basedpyright-checked-blue)](https://github.com/DetachHead/basedpyright)

> 專為 Python 設計的輕量、全功能數學與物理計算函式庫（Comprehensive Mathematic and Physical Functions Library for Python）。

FuncPie 堅持**零第三方外部依賴（Pure Python Standard Library）**，開箱即用、完全型別標註（Type-Annotated），並原生支援優雅的流式函數管道（Functional Pipelines）。

---

## 核心特色 (Key Features)

- **零外部依賴 (Zero Dependencies)**：僅依賴 Python 原生標準庫，極致輕量、隨裝即用，跨平台一致。
- **全面數學模組 (Comprehensive Mathematics)**：
  - **基礎運算 (Basic & Chaining)**：Calculation 鏈式與累加計算、安全運算。
  - **代數與數論 (Algebra & Number Theory)**：多項式微積分與求根、GCD/LCM、質數篩法、質因數分解、排列組合。
  - **微積分與數值分析 (Calculus & Numerical)**：數值微分 (前向/後向/中心差分)、高精度數值積分 (辛普森法/梯形法)、牛頓法與二分法求根、多元梯度。
  - **線性代數 (Linear Algebra)**：2D/3D 幾何向量、矩陣乘法、行列式、高斯消去求逆、線性方程組求解 ($Ax = b$)。
  - **統計與機率 (Statistics & Probability)**：敘述統計、百分位數、四分位數、偏度、峰度、皮爾森相關係數、最小平方法線性迴歸、常態/二項/卜瓦松分布。
  - **幾何與座標轉換 (Geometry & Coordinates)**：多邊形鞋帶公式面積、2D/3D 立體幾何、極座標/球座標/柱座標轉換。
  - **特殊函數與激活函數 (Special & Activations)**：Gamma、Beta、Erf、Sigmoid、ReLU、Leaky ReLU、Softmax。
- **完整物理模組 (Full Physics Suite)**：
  - **精確物理常數 (CODATA Constants)**：光速 $c$、萬有引力常數 $G$、普朗克常數 $h$、波茲曼常數 $k_B$ 等。
  - **古典力學 (Classical Mechanics)**：運動學方程式、拋體軌跡計算、萬有引力、宇宙逃逸速度、動能/功/動量、彈性/非彈性碰撞、單擺與彈簧諧振。
  - **熱力學 (Thermodynamics)**：理想氣體方程式 ($PV=nRT$)、熱傳導、熱膨脹、卡諾循環效率、斯特凡-波茲曼黑體輻射、維恩位移定律。
  - **電磁學與電路 (Electromagnetism & Circuits)**：庫侖定律、歐姆定律、電阻/電容串並聯、RC 電路充放電時態、勞侖茲力、螺線管磁場。
  - **波動與光學 (Waves & Optics)**：聲速溫變、都卜勒效應、司乃耳定律、全反射臨界角、薄透鏡成像方程式、分貝聲強級。
  - **近代物理 (Modern & Quantum Physics)**：狹義相對論勞侖茲因子 $\gamma$、時間膨脹、長度收縮、質能等價 ($E=mc^2$)、光子能量、物質波波長、放射性衰變期。
  - **單位轉換器 (Unit Converter)**：長度、質量、時間、壓力、能量、速度、角度與溫標換算。
- **流式函數管道 (Functional Pipelines)**：支援 Unix 風格的 `value | f | g` 管道與 `pipe()`, `compose()`, `curry()`。
- **向後相容 (Backward Compatible)**：完全相容原先 `libs.math.basic.Calculation` 與 `func_pie` 引用。

---

## 安裝方式 (Installation)

### 標準安裝 (從源碼安裝)
```bash
git clone https://github.com/ai-tech-tw/func_pie.git
cd func_pie
pip install .
```

### 開發者模式安裝 (Editable Mode)
```bash
git clone https://github.com/ai-tech-tw/func_pie.git
cd func_pie
pip install -e .
```

---

## 快速上手 (Quickstart)

### 1. 基礎計算與鏈式運算 (Calculation & Pipelines)
```python
from funcpie import Calculation
from funcpie.core import pipeable

# 支援傳統介面與鏈式調用
calc = Calculation(10)
calc.add(15).mul(2).sub(10).div(2)
print(calc.value)  # ((10 + 15) * 2 - 10) / 2 = 20.0

# Unix 管道運算風格
@pipeable
def add_ten(x):
    return x + 10

@pipeable
def square(x):
    return x ** 2

result = 5 | add_ten | square
print(result)  # (5 + 10)^2 = 225
```

### 2. 向量與線性代數 (Linear Algebra)
```python
from funcpie.math import Vector2D, Matrix

# 2D 向量
v1 = Vector2D(3, 4)
v2 = Vector2D(1, 2)
print(v1.magnitude)   # 5.0
print(v1.dot(v2))     # 11.0

# 矩陣與聯立方程式求解
# 2x +  y = 4
#  x + 3y = 7
A = Matrix([[2, 1], [1, 3]])
x = A.solve([4, 7])
print(x)  # [1.0, 2.0]
```

### 3. 微積分與數值分析 (Calculus & Numerical)
```python
from funcpie.math import derivative, integrate, find_root_newton

# 數值微分：f(x) = x^3 在 x = 2 的導數
df = derivative(lambda x: x ** 3, 2.0)  # 約 12.0

# 數值積分：f(x) = x^2 在 [0, 3] 的定積分
area = integrate(lambda x: x ** 2, 0.0, 3.0)  # 9.0

# 牛頓法求根：x^2 - 4 = 0
root = find_root_newton(lambda x: x ** 2 - 4.0, x0=3.0)  # 2.0
```

### 4. 古典力學與拋體模擬 (Mechanics)
```python
from funcpie.physics import mechanics

# 拋體運動分析 (初速 50 m/s，發射角 45 度)
summary = mechanics.projectile_motion(v0=50.0, angle_deg=45.0)
print(f"最大飛行高度: {summary.max_height:.2f} m")
print(f"總滯空時間:   {summary.flight_time:.2f} s")
print(f"水平射程:     {summary.horizontal_range:.2f} m")

# 地球逃逸速度
v_esc = mechanics.escape_velocity(mass=5.972e24, radius=6.371e6)
print(f"地球逃逸速度: {v_esc / 1000.0:.2f} km/s")
```

### 5. 近代物理與相對論 (Modern Physics)
```python
from funcpie.physics import constants, modern

# 0.8 倍光速下的勞侖茲因子與時間膨脹
v = 0.8 * constants.c
gamma = modern.lorentz_factor(v)
dilated_time = modern.time_dilation(proper_time=1.0, velocity=v)
print(f"勞侖茲因子: {gamma:.4f}")
print(f"膨脹後的觀測時間: {dilated_time:.4f} 秒")

# 質能等價 (1 kg 靜止質量能量)
print(f"E = mc^2: {modern.rest_energy(1.0):.2e} J")
```

### 6. 單位轉換 (Unit Conversions)
```python
from funcpie.physics import units

print(units.convert(100, "km/h", "mph"))         # 62.14 mph
print(units.celsius_to_fahrenheit(100))          # 212.0 °F
print(units.convert(1, "atm", "psi"))            # 14.70 psi
print(units.convert(1, "light_year", "km"))      # 9.46e12 km
```

---

## 專案結構 (Directory Structure)

```
func_pie/
├── funcpie/                    # 主要套件目錄
│   ├── __init__.py             # 核心匯出與版本號
│   ├── py.typed                # PEP 561 型別支援標記
│   ├── core/                   # 核心工具
│   │   ├── __init__.py
│   │   └── pipeline.py         # 流式管道 (pipe, compose, curry, Pipeable)
│   ├── math/                   # 數學模組
│   │   ├── __init__.py
│   │   ├── basic.py            # 基本算術與安全運算
│   │   ├── algebra.py          # 多項式、方程求根、數論、排列組合
│   │   ├── calculus.py         # 數值微分、數值積分、求根法
│   │   ├── linalg.py           # 2D/3D 向量、矩陣與線性方程求解
│   │   ├── statistics.py       # 敘述統計、迴歸分析、機率分布
│   │   ├── geometry.py         # 2D/3D 幾何圖形與座標轉換
│   │   └── special.py          # Gamma, Beta, Erf, 激活函數
│   └── physics/                # 物理模組
│       ├── __init__.py
│       ├── constants.py        # CODATA 國際推薦物理常數
│       ├── mechanics.py        # 運動學、動力學、重力、碰撞、諧振
│       ├── thermodynamics.py   # 理想氣體、熱傳導、黑體輻射、熱機效率
│       ├── electromagnetism.py # 庫侖定律、電路分析、電磁力
│       ├── waves.py            # 波動、光學折射、薄透鏡、聲學
│       ├── modern.py           # 相對論、光電效應、衰變定律
│       └── units.py            # 跨領域物理單位轉換器
├── func_pie/                   # 命名相容別名模組
├── libs/                       # 原版相容模組 (libs.math.basic)
├── tests/                      # 完整單元測試 (44+ 個測試項)
├── examples/                   # 範例腳本
│   ├── quickstart.py
│   └── physics_simulation.py
├── pyproject.toml              # PEP 517/621 建置設定
├── setup.py                    # 相容性安裝檔
└── LICENSE                     # MIT License
```

---

## 測試與型別檢查 (Running Tests & Type Checking)

### 執行單元測試
使用 `pytest` 執行完整測試套件：
```bash
pytest -v
```
亦可使用 Python 原生內建測試執行器：
```bash
python3 -m unittest discover tests -v
```

### 靜態型別檢查
使用 `basedpyright` 進行嚴格型別檢查：
```bash
npx basedpyright
```

### 執行範例程式
```bash
python3 examples/quickstart.py
python3 examples/physics_simulation.py
```

---

## 授權條款 (License)

本專案採用 [MIT License](LICENSE) 授權。
