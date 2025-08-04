# import matplotlib.pyplot as plt

# # 数据准备
# steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# score_both = [0.89569, 0.92076, 0.93972, 0.93930, 0.94514, 0.94930, 0.95097, 0.95472, 0.95472, 0.95472]
# score_text = [0.91132, 0.92382, 0.93436, 0.93645, 0.93853, 0.93853, 0.94238, 0.94238, 0.94624, 0.94999]
# score_image = [0.87416, 0.91710, 0.92470, 0.92804, 0.93189, 0.93148, 0.93700, 0.93700, 0.93700, 0.93700]

# # 绘图
# plt.figure(figsize=(6, 4))
# plt.plot(steps, score_both, marker='o', label='Both')
# plt.plot(steps, score_text, marker='s', label='Text')
# plt.plot(steps, score_image, marker='^', label='Image')

# # 图表设置
# plt.xlabel("Optimization Steps")
# plt.ylabel("Geneval Score")
# plt.xticks(steps)
# plt.ylim(0.86, 0.96)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()

# # 保存为高清 PDF（可选）
# plt.savefig("optimization_score_comparison.pdf", dpi=500, bbox_inches='tight', format='pdf')
# plt.show()

import matplotlib.pyplot as plt

# 设置字体与样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 数据
steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# score_both = [0.89569, 0.92076, 0.93972, 0.93930, 0.94514, 0.94930, 0.95097, 0.95472, 0.95472, 0.95472]
score_both = [0.89569, 0.92076, 0.93972, 0.94183, 0.94514, 0.94930, 0.95097, 0.95472, 0.95472, 0.95472]
score_text = [0.91132, 0.92382, 0.93436, 0.93645, 0.93853, 0.93853, 0.94238, 0.94238, 0.94624, 0.94999]
# score_image = [0.87416, 0.91710, 0.92470, 0.92804, 0.93189, 0.93148, 0.93700, 0.93700, 0.93700, 0.93700]
score_image = [0.87416, 0.91710, 0.92470, 0.92804, 0.93189, 0.93248, 0.93700, 0.93700, 0.93700, 0.93700]

# 绘图
plt.figure(figsize=(6, 4))
plt.plot(steps, score_both, marker='o', linestyle='-', color='#ff7f0e', label='Both')
plt.plot(steps, score_text, marker='s', linestyle='--', color='#1f77b4', label='Text')
plt.plot(steps, score_image, marker='^', linestyle='-.', color='#2ca02c', label='Image')

# 美化
plt.xlabel("Optimization Steps")
plt.ylabel("Geneval Score")
plt.xticks(steps)
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96])
plt.ylim(0.86, 0.96)
plt.grid(True, linestyle='--', alpha=0.5)

# 去除顶部和右边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
plt.legend(loc='lower right', frameon=True)

# 保存
plt.tight_layout()
plt.savefig("step_scaling.pdf", dpi=600, bbox_inches='tight', format='pdf')
plt.show()
