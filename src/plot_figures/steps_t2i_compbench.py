import matplotlib.pyplot as plt

# 字体与样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 数据
steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# score_both = [0.4695, 0.4896, 0.4934, 0.5048, 0.5164, 0.5164, 0.5223, 0.5248, 0.5309, 0.5371]
score_both = [0.4695, 0.4896, 0.4934, 0.5048, 0.5164, 0.5164, 0.5223, 0.5248, 0.5309, 0.5325]
score_image = [0.4458, 0.4634, 0.4744, 0.4806, 0.4859, 0.4901, 0.4972, 0.5016, 0.5028, 0.5043]
score_text  = [0.4569, 0.4702, 0.4842, 0.4892, 0.4982, 0.5058, 0.5080, 0.5147, 0.5188, 0.5210]

# 绘图
plt.figure(figsize=(6, 4))

# 线型、标记、颜色与原图一致
plt.plot(steps, score_both,  marker='o', linestyle='-',  color='#ff7f0e', label='Both')
plt.plot(steps, score_text,  marker='s', linestyle='--', color='#1f77b4', label='Text')
plt.plot(steps, score_image, marker='^', linestyle='-.', color='#2ca02c', label='Image')

# 轴标签与刻度
plt.xlabel("Optimization Steps")
plt.ylabel("T2I-CompBench Score")
plt.xticks(steps)

# 根据数据范围设置刻度与范围（与原图风格对应的固定刻度）
plt.yticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54])
plt.ylim(0.44, 0.54)

# 网格与边框
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
plt.legend(loc='lower right', frameon=True)

# 保存与展示
plt.tight_layout()
plt.savefig("step_t2i_compbench.pdf", dpi=600, bbox_inches='tight', format='pdf')
plt.show()
