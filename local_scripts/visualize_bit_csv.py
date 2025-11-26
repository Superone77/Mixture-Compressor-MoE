import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import ast
import numpy as np

# 1. 读 CSV
# 假设文件名是 data.csv，你可以改成自己的
df = pd.read_csv("/Users/superone77/Code/Mixture-Compressor-MoE/data/alpha_2_5bit.csv")

# 2. 解析 bit 字段（从字符串变成 Python list）
df["bit_list"] = df["bit"].apply(ast.literal_eval)

# 3. 计算数值范围，用来做归一化（决定明暗）
all_vals = [v for bits in df["bit_list"] for v in bits]
vmin = min(all_vals)
vmax = max(all_vals)

# 避免除零
if vmax == vmin:
    vmax = vmin + 1e-6

def normalize(val):
    return (val - vmin) / (vmax - vmin)

cmap = plt.cm.viridis  # 彩色色图：数值小 -> 深蓝/紫，数值大 -> 黄/绿

# 4. 准备画布
fig, ax = plt.subplots(figsize=(12, 6))

# 为了方便设置坐标轴范围
layers = sorted(df["layer"].unique())
experts = sorted(df["expert_id"].unique())

# 5. 逐个格子画 3 个横向小长方形
for _, row in df.iterrows():
    layer = row["layer"]
    expert = row["expert_id"]
    bits = row["bit_list"]  # 长度应该为 3

    # 每个大格子的宽度 = 1，把它在 x 方向等分成 3 份
    cell_x = layer
    cell_y = expert
    cell_w = 1.0
    cell_h = 1.0
    sub_w = cell_w / 3.0

    for i, val in enumerate(bits):
        # 每个小条的左下角坐标
        left = cell_x + i * sub_w
        bottom = cell_y

        color = cmap(normalize(val))

        rect = patches.Rectangle(
            (left, bottom),
            sub_w,
            cell_h,
            facecolor=color,
            edgecolor="none"
        )
        ax.add_patch(rect)

# 6. 设置坐标轴范围和刻度
ax.set_xlim(min(layers), max(layers) + 1)
ax.set_ylim(min(experts), max(experts) + 1)

# 让 y 轴从上到下或者从下到上都可以，看你习惯：
# 如果希望 expert_id 从上往下递增，可以反转 y 轴：
ax.invert_yaxis()

ax.set_xticks([l + 0.5 for l in layers])
ax.set_xticklabels(layers)
ax.set_yticks([e + 0.5 for e in experts])
ax.set_yticklabels(experts)

ax.set_xlabel("layer_id")
ax.set_ylabel("expert_id")
ax.set_title("Bit Heatmap (3 horizontal strips per cell)")

ax.set_aspect("equal")  # 保持方格

# 7. 加一个 colorbar 表示数值大小
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("bit value")

plt.tight_layout()
plt.show()
