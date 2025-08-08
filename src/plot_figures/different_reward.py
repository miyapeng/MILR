# import matplotlib.pyplot as plt

# # 设置字体与样式（ICLR风格）
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 12,
# })

# # 数据
# steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# reward_scores = {
#     "Self Reward": [0.78534] * 10,
#     "GPT4-o": [0.83196, 0.83706, 0.83915, 0.83706, 0.83706, 0.84123, 0.83915, 0.83706, 0.83915, 0.84123],
#     "UnifiedReward": [0.81266, 0.83192, 0.82766, 0.82620, 0.82828, 0.83547, 0.83432, 0.84213, 0.84088, 0.84182],
#     "MixedReward": [0.82045, 0.84668, 0.85043, 0.85449, 0.85429, 0.85418, 0.85835, 0.85877, 0.86043, 0.86710],
#     "SANA": [0.85236, 0.85778, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153],
# }

# # 颜色与marker设置（ICLR风格配色）
# colors = ['dimgray', 'royalblue', 'seagreen', 'crimson', 'indigo']
# markers = ['o', 's', 'D', '^', 'v']

# # 绘图
# plt.figure(figsize=(7, 4.5))

# for i, (label, scores) in enumerate(reward_scores.items()):
#     plt.plot(steps, scores, label=label, marker=markers[i], color=colors[i], linewidth=1.8)

# # 坐标与标题美化
# plt.xlabel("Optimization Steps")
# plt.ylabel("Geneval Score")
# plt.ylim(0.78, 0.87)
# plt.xticks(steps)
# plt.grid(True, linestyle='--', alpha=0.6)

# # 图例美化：放下方，不遮挡主图
# plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

# # 紧凑布局
# plt.tight_layout()

# # 保存
# plt.savefig("different_rewards.pdf", dpi=500, bbox_inches='tight', format='pdf')
# plt.show()

import matplotlib.pyplot as plt

# 设置字体与样式（ICLR风格）
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 数据
steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
reward_scores = {
    "Self Reward": [0.78534] * 10,
    "GPT4-o": [0.83196, 0.83706, 0.83915, 0.83706, 0.83706, 0.84123, 0.83915, 0.83706, 0.83915, 0.84123],
    "UnifiedReward": [0.81266, 0.83192, 0.82766, 0.82620, 0.82828, 0.83547, 0.83432, 0.84213, 0.84088, 0.84182],
    "MixedReward": [0.82045, 0.84668, 0.85043, 0.85449, 0.85429, 0.85418, 0.85835, 0.85877, 0.86043, 0.86710],
    "SANA": [0.85236, 0.85778, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153, 0.86153],
}

# 改进的颜色和 marker 设置
colors = ['seagreen', 'royalblue', 'darkorange', 'crimson', 'indigo']
markers = ['o', 's', 'D', '^', 'v']

# 绘图
plt.figure(figsize=(7, 4.5))

for i, (label, scores) in enumerate(reward_scores.items()):
    plt.plot(steps, scores, label=label, marker=markers[i], color=colors[i], linewidth=1.8)

# 坐标与标题美化
plt.xlabel("Optimization Steps")
plt.ylabel("Geneval Score")
plt.ylim(0.78, 0.87)
plt.xticks(steps)
plt.grid(True, linestyle='--', alpha=0.6)

# 去除上右边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例美化：放下方，不遮挡主图
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

# 紧凑布局
plt.tight_layout()

# 保存图像
plt.savefig("different_rewards.pdf", dpi=500, bbox_inches='tight', format='pdf')
plt.show()
