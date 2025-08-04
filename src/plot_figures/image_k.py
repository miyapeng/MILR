import matplotlib.pyplot as plt

# 设置字体与样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 设置数据
image_k = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
score_5 =  [0.9303, 0.9280, 0.9151, 0.9204, 0.9194, 0.9127, 0.9090, 0.9126, 0.9082, 0.9082]
score_10 = [0.9444, 0.9450, 0.9323, 0.9309, 0.9307, 0.9268, 0.9239, 0.9286, 0.9300, 0.9241]
score_20 = [0.9489, 0.9530, 0.9369, 0.9413, 0.9388, 0.9349, 0.9357, 0.9307, 0.9359, 0.9346]

# 创建图表
plt.figure(figsize=(6, 4))
plt.plot(image_k, score_5, marker='o', label='5 steps')
plt.plot(image_k, score_10, marker='s', label='10 steps')
plt.plot(image_k, score_20, marker='^', label='20 steps')

# 图表设置
plt.xlabel("Image Optimization Ratio")
plt.ylabel("Geneval Score")
plt.ylim(0.9, 0.955)
plt.xticks(image_k)
plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(loc='lower left')
plt.legend()
plt.tight_layout()

# 保存高清 PDF 图像
plt.savefig("geneval_imagek.pdf", dpi=500, bbox_inches='tight', format='pdf')
plt.show()
