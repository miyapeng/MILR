import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ===== 可调参数 =====
FIG_W, FIG_H = 15.0, 4.9     # 画布尺寸
WSPACE = 0.31                # 子图之间的横向间距
HSPACE = 0.33                # 子图与图例行之间的纵向间距
LEGEND_ROW_RATIO = 0.17      # 图例行高度占比（调大→更高）
LABELPAD_X = 4               # x轴标签与刻度距离
LABELPAD_Y = 4               # y轴标签与刻度距离

# ===== 字号规范 =====
SIZE_TICK   = 12   # 坐标轴刻度数字
SIZE_LABEL  = 13   # 轴名称：Steps / Performance
SIZE_TITLE  = 15   # 子图标题：GenEval / T2I-CompBench / WISE
SIZE_PANEL  = 16   # 面板标注：(a)(b)(c)
SIZE_LEGEND = 13   # 图例

# plt.rcParams.update({"font.family": "serif", "font.size": 13})
plt.rcParams.update({
    "font.family": "serif",
    "font.size": SIZE_TICK,         # 作为刻度的默认字号
    "axes.labelsize": SIZE_LABEL,   # 轴名称字号
    "axes.titlesize": SIZE_TITLE,   # 标题字号
    "xtick.labelsize": SIZE_TICK,   # x 轴刻度
    "ytick.labelsize": SIZE_TICK,   # y 轴刻度
    "legend.fontsize": SIZE_LEGEND, # 图例字号
})

steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# —— GenEval ——
both_g = [0.89569, 0.92076, 0.93972, 0.94183, 0.94514, 0.94930, 0.95097, 0.95472, 0.95472, 0.95472]
text_g = [0.91132, 0.92382, 0.93436, 0.93645, 0.93853, 0.93853, 0.94238, 0.94238, 0.94624, 0.94999]
img_g  = [0.87416, 0.91710, 0.92470, 0.92804, 0.93189, 0.93248, 0.93700, 0.93700, 0.93700, 0.93700]

# —— T2I-CompBench ——
both_t = [0.4695, 0.4896, 0.4934, 0.5048, 0.5164, 0.5164, 0.5223, 0.5248, 0.5309, 0.5325]
text_t = [0.4569, 0.4702, 0.4842, 0.4892, 0.4982, 0.5058, 0.5080, 0.5147, 0.5188, 0.5210]
img_t  = [0.4458, 0.4634, 0.4744, 0.4806, 0.4859, 0.4901, 0.4972, 0.5016, 0.5028, 0.5043]

# —— WISE ——
both_w = [0.59, 0.60, 0.61, 0.61, 0.62, 0.62, 0.62, 0.63, 0.63, 0.63]
text_w = [0.57, 0.57, 0.58, 0.59, 0.59, 0.60, 0.60, 0.61, 0.61, 0.61]
img_w  = [0.53, 0.54, 0.55, 0.55, 0.55, 0.55, 0.55, 0.56, 0.56, 0.56]

# ===== 上 1 行放子图，下 1 行放图例 =====
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(
    nrows=2, ncols=3, figure=fig,
    height_ratios=[1.0, LEGEND_ROW_RATIO],
    width_ratios=[1, 1, 1],
    wspace=WSPACE, hspace=HSPACE
)

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis("off")

def style(ax, title, ylim, yticks):
    ax.set_title(title, pad=4)
    ax.set_xlabel("Steps", labelpad=LABELPAD_X)
    ax.set_ylabel("Performance", labelpad=LABELPAD_Y)
    ax.set_xticks(steps)
    if ylim:   ax.set_ylim(*ylim)
    if yticks: ax.set_yticks(yticks)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 左：GenEval
l1, = axes[0].plot(steps, both_g, marker='o', linestyle='-',  color='#ff7f0e', label='Both')
l2, = axes[0].plot(steps, text_g, marker='s', linestyle='--', color='#1f77b4', label='Text')
l3, = axes[0].plot(steps, img_g,  marker='^', linestyle='-.', color='#2ca02c', label='Image')
style(axes[0], "GenEval", (0.86, 0.96), [0.86, 0.88, 0.90, 0.92, 0.94, 0.96])

# 中：T2I-CompBench
axes[1].plot(steps, both_t, marker='o', linestyle='-',  color='#ff7f0e')
axes[1].plot(steps, text_t, marker='s', linestyle='--', color='#1f77b4')
axes[1].plot(steps, img_t,  marker='^', linestyle='-.', color='#2ca02c')
style(axes[1], "T2I-CompBench", (0.44, 0.54), [0.44, 0.46, 0.48, 0.50, 0.52, 0.54])
# axes[1].set_title("T2I-CompBench", y=0.95, pad=0)

# 右：WISE
axes[2].plot(steps, both_w, marker='o', linestyle='-',  color='#ff7f0e')
axes[2].plot(steps, text_w, marker='s', linestyle='--', color='#1f77b4')
axes[2].plot(steps, img_w,  marker='^', linestyle='-.', color='#2ca02c')
style(axes[2], "WISE", (0.52, 0.64), [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64])


panel_labels = ['(a)', '(b)', '(c)']
for ax, lab in zip(axes, panel_labels):
    ax.text(-0.18, 1.04, lab, transform=ax.transAxes,  # x 越小越靠左；y 越大越靠上
            ha='right', va='bottom', fontsize=16, clip_on=False)
# 如果觉得太紧：HSPACE = 0.36 或 FIG_H = 4.9


# 统一底部图例（在独立 legend_ax，绝不与坐标轴重叠）
legend_ax.legend(
    handles=[l1, l2, l3], labels=['MILR', 'w/o Image', 'w/o Text'],
    loc='center', ncol=3, frameon=True
)

fig.savefig("test_time_scaling_horizontal.pdf", dpi=600, bbox_inches='tight')
plt.show()
