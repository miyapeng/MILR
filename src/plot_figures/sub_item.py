import matplotlib.pyplot as plt
import numpy as np

# 设置字体与样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 数据
steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
categories = {
    "Single Obj.": [1.0] * len(steps),
    "Two Obj.": [0.9192, 0.9596, 0.9596, 0.9596, 0.9596, 0.9596, 0.9596, 0.9596, 0.9596, 0.9596],
    "Counting": [0.7375, 0.7875, 0.85, 0.85, 0.8625, 0.8875, 0.8875, 0.9, 0.9, 0.9],
    "Color": [0.9574, 0.9574, 0.9787, 0.9787, 0.9787, 0.9787, 0.9787, 0.9787, 0.9787, 0.9787],
    "Pos.": [0.92, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98],
    "Color Attri.": [0.84, 0.87, 0.87, 0.88, 0.89, 0.89, 0.90, 0.91, 0.91, 0.91],
}

# 颜色列表
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

# 设置子图
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
axs = axs.flatten()

for idx, (cat, values) in enumerate(categories.items()):
    axs[idx].plot(steps, values, marker='o', color=colors[idx], label=cat)
    axs[idx].set_title(cat, fontsize=11)
    axs[idx].set_xlabel("Steps")
    axs[idx].set_ylabel("Score")
    axs[idx].grid(True, linestyle='--', alpha=0.6)
    axs[idx].set_ylim(0.7, 1.01)

# 🔧 去除每个子图的右侧和上侧边框
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("geneval_attribute_curve.pdf", dpi=500, bbox_inches='tight', format='pdf')
plt.show()
