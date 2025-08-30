import matplotlib.pyplot as plt

# 设置字体与样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# 设置数据
text_k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
score_5 =  [0.923487, 0.942670, 0.928497, 0.930507, 0.937590, 0.934553, 0.924100, 0.931613, 0.926530, 0.933327]
score_10 = [0.933917, 0.956710, 0.941350, 0.939957, 0.950373, 0.943447, 0.932090, 0.939607, 0.941403, 0.945830]
score_20 = [0.943020, 0.961297, 0.944683, 0.944473, 0.953850, 0.951223, 0.938827, 0.947593, 0.953907, 0.952567]

# 开始画图
plt.figure(figsize=(6, 4))  # 控制图大小（单位：英寸）
plt.plot(text_k, score_5, marker='o', label='5 steps')
plt.plot(text_k, score_10, marker='s', label='10 steps')
plt.plot(text_k, score_20, marker='^', label='20 steps')

# 图表修饰
plt.xlabel("Text Optimization Ratio")
plt.ylabel("Geneval Score")
plt.ylim(0.92, 0.97)  # 根据分数区间微调Y轴范围
plt.xticks(text_k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
# plt.legend(loc='lower left')
plt.tight_layout()

# 保存为高清 PDF
plt.savefig("geneval_textk.pdf", dpi=500, bbox_inches='tight', format='pdf')
plt.show()