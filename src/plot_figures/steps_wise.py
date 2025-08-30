import matplotlib.pyplot as plt

# 字体与样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 数据
steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

score_both = [0.59, 0.60, 0.61, 0.61, 0.62, 0.62, 0.62, 0.63, 0.63, 0.63]
score_text = [0.57, 0.57, 0.58, 0.59, 0.59, 0.60, 0.60, 0.61, 0.61, 0.61]
score_image = [0.53, 0.54, 0.55, 0.55, 0.55, 0.55, 0.55, 0.56, 0.56, 0.56]

# 绘图
plt.figure(figsize=(6, 4))

# 线型、标记、颜色保持一致
plt.plot(steps, score_both,  marker='o', linestyle='-',  color='#ff7f0e', label='Both')
plt.plot(steps, score_text,  marker='s', linestyle='--', color='#1f77b4', label='Text')
plt.plot(steps, score_image, marker='^', linestyle='-.', color='#2ca02c', label='Image')

# 轴标签与刻度
plt.xlabel("Optimization Steps")
plt.ylabel("WISE Score")
plt.xticks(steps)

# y 轴范围与刻度（根据数据调整到 0.52–0.64）
plt.yticks([0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64])
plt.ylim(0.52, 0.64)

# 网格与边框
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
plt.legend(loc='lower right', frameon=True)

# 保存与展示
plt.tight_layout()
plt.savefig("step_wise.pdf", dpi=600, bbox_inches='tight', format='pdf')
plt.show()
