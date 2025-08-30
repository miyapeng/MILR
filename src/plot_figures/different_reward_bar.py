# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# categories = ["Single Obj.", "Two Obj.", "Counting", "Colors", "Position", "Color Attri.", "Overall"]
# rewards = {
#     "Self Reward": [1.00, 0.90, 0.50, 0.88, 0.77, 0.65, 0.79],
#     "GPT4-o": [0.98, 0.94, 0.80, 0.88, 0.77, 0.68, 0.84],
#     "Unified Reward": [1.00, 0.91, 0.76, 0.89, 0.81, 0.66, 0.85],
#     "SANA": [1.00, 0.93, 0.68, 0.91, 0.88, 0.77, 0.86],
#     "Mixed Reward": [1.00, 0.90, 0.81, 0.89, 0.88, 0.69, 0.87],
#     "Metric Reward": [1.00, 0.96, 0.90, 0.98, 0.98, 0.91, 0.95],
# }

# # 颜色：沿用你之前示例的深色系
# colors = ['seagreen', 'royalblue', 'darkorange', 'crimson', 'indigo', 'darkcyan']

# x = np.arange(len(categories))
# width = 0.13

# plt.figure(figsize=(9, 5))

# # 绘制每组柱子
# for i, (label, scores) in enumerate(rewards.items()):
#     plt.bar(x + i*width - (len(rewards)/2 - 0.5)*width, scores, width=width, label=label, color=colors[i])

# # 美化坐标轴
# plt.xticks(x, categories, rotation=25, ha='right')
# plt.ylabel("Geneval Score")
# plt.ylim(0.45, 1.05)

# # 去除上右边框
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # 不画网格
# plt.grid(False)

# # 图例放下方
# plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# categories = ["Single Obj.", "Two Obj.", "Counting", "Colors", "Position", "Color Attri.", "Overall"]
# rewards = ["Self Reward", "GPT4-o", "Unified Reward", "SANA", "Mixed Reward", "Metric Reward"]
# data = np.array([
#     [1.00, 0.90, 0.50, 0.88, 0.77, 0.65, 0.79],
#     [0.98, 0.94, 0.80, 0.88, 0.77, 0.68, 0.84],
#     [1.00, 0.91, 0.76, 0.89, 0.81, 0.66, 0.85],
#     [1.00, 0.93, 0.68, 0.91, 0.88, 0.77, 0.86],
#     [1.00, 0.90, 0.81, 0.89, 0.88, 0.69, 0.87],
#     [1.00, 0.96, 0.90, 0.98, 0.98, 0.91, 0.95]
# ])

# # 设置样式（新的颜色方案，明亮但不柔和）
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
# })
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# # 柱状图参数
# x = np.arange(len(categories))
# bar_width = 0.13

# plt.figure(figsize=(8, 4.5))

# for i, (reward, color) in enumerate(zip(rewards, colors)):
#     plt.bar(x + i * bar_width, data[i], width=bar_width, label=reward, color=color)

# # 坐标轴与标签
# plt.ylabel("Geneval Score")
# plt.xticks(x + bar_width * (len(rewards) - 1) / 2, categories, rotation=20, ha='right')

# # 去除上右边框，去掉背景网格
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.grid(False)

# # 图例放下方
# plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

# # 紧凑布局
# plt.tight_layout()

# # 保存并显示
# plt.savefig("geneval_bar_chart.pdf", dpi=500, bbox_inches='tight', format='pdf')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 类别（列）
categories = ["Single Obj.", "Two Obj.", "Counting", "Colors", "Position", "Color Attri.", "Overall"]

# 方法顺序：把 MixedReward 放到 SANA 后面
rewards = ["Self Reward", "GPT4-o", "UnifiedReward", "SANA", "MixedReward", "Oracle Reward"]
# 颜色与顺序一一对应（Oracle 用灰色）
colors  = ["seagreen", "royalblue", "darkorange", "indigo", "crimson", "#7f7f7f"]

# 数据按上述顺序对应行
data = np.array([
    [1.00, 0.90, 0.50, 0.88, 0.77, 0.65, 0.79],  # Self Reward
    [0.98, 0.94, 0.80, 0.88, 0.77, 0.68, 0.84],  # GPT4-o
    [1.00, 0.92, 0.76, 0.89, 0.81, 0.66, 0.84],  # UnifiedReward
    [1.00, 0.93, 0.68, 0.91, 0.88, 0.77, 0.86],  # SANA
    [1.00, 0.90, 0.83, 0.89, 0.88, 0.75, 0.87],  # MixedReward
    [1.00, 0.96, 0.90, 0.98, 0.98, 0.91, 0.95],  # Oracle Reward (原 Metric Reward)
])

# 全局样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
})

# 柱状图布局
x = np.arange(len(categories))
n_series = len(rewards)
bar_width = 0.13

plt.figure(figsize=(8, 4.5))

# 绘制
for i, (reward, color) in enumerate(zip(rewards, colors)):
    plt.bar(x + (i - (n_series - 1) / 2) * bar_width, data[i], width=bar_width, label=reward, color=color)

# 轴与标签
plt.ylabel("GenEval Score")
plt.xticks(x, categories, rotation=20, ha="right")

# 去掉上/右边框 & 关闭网格
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(False)

# 图例置底
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig("geneval_bar_chart_rewards.pdf", dpi=500, bbox_inches="tight", format="pdf")
plt.show()