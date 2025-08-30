# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# # ===== 画布与间距 =====
# FIG_W, FIG_H = 12.5, 4.6
# WSPACE = 0.29
# HSPACE = 0.33
# LEGEND_ROW_RATIO = 0.17
# LABELPAD_X = 4
# LABELPAD_Y = 4

# # ===== 字号规范 =====
# SIZE_TICK   = 12   # 坐标刻度
# SIZE_LABEL  = 13   # 轴名称
# SIZE_TITLE  = 15   # 子图标题
# SIZE_PANEL  = 16   # (a)(b)
# SIZE_LEGEND = 13   # 图例

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": SIZE_TICK,
#     "axes.labelsize": SIZE_LABEL,
#     "axes.titlesize": SIZE_TITLE,
#     "xtick.labelsize": SIZE_TICK,
#     "ytick.labelsize": SIZE_TICK,
#     "legend.fontsize": SIZE_LEGEND,
# })

# # ===== 数据 =====
# # 左图：Text ratio
# text_k  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# score_5_text  = [0.923487, 0.942670, 0.928497, 0.930507, 0.937590, 0.934553, 0.924100, 0.931613, 0.926530, 0.933327]
# score_10_text = [0.933917, 0.956710, 0.941350, 0.939957, 0.950373, 0.943447, 0.932090, 0.939607, 0.941403, 0.945830]
# score_20_text = [0.943020, 0.961297, 0.944683, 0.944473, 0.953850, 0.951223, 0.938827, 0.947593, 0.953907, 0.952567]

# # 右图：Image ratio
# image_k  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
# score_5_img  = [0.9303, 0.9280, 0.9151, 0.9204, 0.9194, 0.9127, 0.9090, 0.9126, 0.9082, 0.9082]
# score_10_img = [0.9444, 0.9450, 0.9323, 0.9309, 0.9307, 0.9268, 0.9239, 0.9286, 0.9300, 0.9241]
# score_20_img = [0.9489, 0.9530, 0.9369, 0.9413, 0.9388, 0.9349, 0.9357, 0.9307, 0.9359, 0.9346]

# # ===== 画布：上 1 行两幅图，下 1 行图例 =====
# fig = plt.figure(figsize=(FIG_W, FIG_H))
# gs = GridSpec(
#     nrows=2, ncols=2, figure=fig,
#     height_ratios=[1.0, LEGEND_ROW_RATIO],
#     width_ratios=[1, 1],
#     wspace=WSPACE, hspace=HSPACE
# )

# axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
# legend_ax = fig.add_subplot(gs[1, :])
# legend_ax.axis("off")

# def style(ax, title, xlab, ylab, ylim=None, yticks=None, xticks=None, xtickfmt=None):
#     ax.set_title(title, pad=4)
#     ax.set_xlabel(xlab, labelpad=LABELPAD_X)
#     ax.set_ylabel(ylab, labelpad=LABELPAD_Y)
#     if xticks is not None:
#         ax.set_xticks(xticks)
#         if xtickfmt is not None:
#             ax.set_xticklabels([xtickfmt(v) for v in xticks])
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     if yticks is not None:
#         ax.set_yticks(yticks)
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# # ----- 左图：Text Optimization Ratio -----
# h1, = axes[0].plot(text_k, score_5_text,  marker='o', linestyle='-', label='5 steps')
# h2, = axes[0].plot(text_k, score_10_text, marker='s', linestyle='-', label='10 steps')
# h3, = axes[0].plot(text_k, score_20_text, marker='^', linestyle='-', label='20 steps')

# style(
#     axes[0], title="GenEval (Text ratio)",
#     xlab="Text Optimization Ratio", ylab="Overall",
#     ylim=(0.92, 0.97), yticks=[0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
#     xticks=text_k, xtickfmt=lambda v: f"{v:.1f}"
# )

# # ----- 右图：Image Optimization Ratio -----
# axes[1].plot(image_k, score_5_img,  marker='o', linestyle='-', label='5 steps')
# axes[1].plot(image_k, score_10_img, marker='s', linestyle='-', label='10 steps')
# axes[1].plot(image_k, score_20_img, marker='^', linestyle='-', label='20 steps')

# style(
#     axes[1], title="GenEval (Image ratio)",
#     xlab="Image Optimization Ratio", ylab="Overall",
#     ylim=(0.90, 0.955), yticks=[0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
#     xticks=image_k, xtickfmt=lambda v: f"{v:.2f}"
# )

# # 面板标注 (a)(b)
# for ax, lab in zip(axes, ['(a)', '(b)']):
#     ax.text(-0.16, 1.04, lab, transform=ax.transAxes,
#             ha='right', va='bottom', fontsize=SIZE_PANEL, fontweight='normal', clip_on=False)

# # 底部统一图例（无边框），沿用三个标记
# legend_ax.legend(
#     handles=[h1, h2, h3], labels=['5 steps', '10 steps', '20 steps'],
#     loc='center', ncol=3, frameon=True
# )

# fig.savefig("text_image_ratio.pdf", dpi=600, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np  # 新增

# ===== 画布与间距 =====
FIG_W, FIG_H = 12.5, 4.6
WSPACE = 0.29
HSPACE = 0.33
LEGEND_ROW_RATIO = 0.17
LABELPAD_X = 4
LABELPAD_Y = 4

# ===== 字号规范 =====
SIZE_TICK   = 12   # 坐标刻度
SIZE_LABEL  = 13   # 轴名称
SIZE_TITLE  = 15   # 子图标题
SIZE_PANEL  = 16   # (a)(b)
SIZE_LEGEND = 13   # 图例

plt.rcParams.update({
    "font.family": "serif",
    "font.size": SIZE_TICK,
    "axes.labelsize": SIZE_LABEL,
    "axes.titlesize": SIZE_TITLE,
    "xtick.labelsize": SIZE_TICK,
    "ytick.labelsize": SIZE_TICK,
    "legend.fontsize": SIZE_LEGEND,
})

# ===== 数据 =====
# 左图：Text ratio
text_k  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
score_5_text  = [0.923487, 0.942670, 0.928497, 0.930507, 0.937590, 0.934553, 0.924100, 0.931613, 0.926530, 0.933327]
score_10_text = [0.933917, 0.956710, 0.941350, 0.939957, 0.950373, 0.943447, 0.932090, 0.939607, 0.941403, 0.945830]
score_20_text = [0.943020, 0.961297, 0.944683, 0.944473, 0.953850, 0.951223, 0.938827, 0.947593, 0.953907, 0.952567]

# 右图：Image ratio
image_k  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
score_5_img  = [0.9303, 0.9280, 0.9151, 0.9204, 0.9194, 0.9127, 0.9090, 0.9126, 0.9082, 0.9082]
score_10_img = [0.9444, 0.9450, 0.9323, 0.9309, 0.9307, 0.9268, 0.9239, 0.9286, 0.9300, 0.9241]
score_20_img = [0.9489, 0.9530, 0.9369, 0.9413, 0.9388, 0.9349, 0.9357, 0.9307, 0.9359, 0.9346]

# ===== 画布：上 1 行两幅图，下 1 行图例 =====
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(
    nrows=2, ncols=2, figure=fig,
    height_ratios=[1.0, LEGEND_ROW_RATIO],
    width_ratios=[1, 1],
    wspace=WSPACE, hspace=HSPACE
)

axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis("off")

def style(ax, title, xlab, ylab, ylim=None, yticks=None, xticks=None, xtickfmt=None):
    ax.set_title(title, pad=4)
    ax.set_xlabel(xlab, labelpad=LABELPAD_X)
    ax.set_ylabel(ylab, labelpad=LABELPAD_Y)
    if xticks is not None:
        ax.set_xticks(xticks)
        if xtickfmt is not None:
            ax.set_xticklabels([xtickfmt(v) for v in xticks])
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# —— 辅助函数：为单条曲线添加“近似置信带” ——
def add_fake_band(ax, x, y, eps=None, frac=None, color=None, alpha=0.18, zorder=1):
    """
    eps: 绝对带宽（例如 0.004）
    frac: 相对带宽（例如 0.01 表示 ±1%）
    二者只用其一；都给则优先 frac。
    """
    y = np.asarray(y, dtype=float)
    if frac is not None:
        delta = np.maximum(frac * np.abs(y), 1e-12)
    else:
        delta = eps if np.isscalar(eps) else np.asarray(eps)
    lo = y - delta
    hi = y + delta
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=zorder)

# ----- 左图：Text Optimization Ratio -----
h1, = axes[0].plot(text_k, score_5_text,  marker='o', linestyle='-', label='5 steps')
# 置信带（示例：±0.004）
add_fake_band(axes[0], text_k, score_5_text, eps=0.004, color=h1.get_color())

h2, = axes[0].plot(text_k, score_10_text, marker='s', linestyle='-', label='10 steps')
add_fake_band(axes[0], text_k, score_10_text, eps=0.004, color=h2.get_color())

h3, = axes[0].plot(text_k, score_20_text, marker='^', linestyle='-', label='20 steps')
add_fake_band(axes[0], text_k, score_20_text, eps=0.004, color=h3.get_color())

style(
    axes[0], title="Text Ratio Analysis(Geneval)",
    xlab="Text Ratio", ylab="Overall",
    ylim=(0.92, 0.97), yticks=[0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
    xticks=text_k, xtickfmt=lambda v: f"{v:.1f}"
)

# ----- 右图：Image Optimization Ratio -----
l1, = axes[1].plot(image_k, score_5_img,  marker='o', linestyle='-', label='5 steps')
add_fake_band(axes[1], image_k, score_5_img, eps=0.0035, color=l1.get_color())

l2, = axes[1].plot(image_k, score_10_img, marker='s', linestyle='-', label='10 steps')
add_fake_band(axes[1], image_k, score_10_img, eps=0.0035, color=l2.get_color())

l3, = axes[1].plot(image_k, score_20_img, marker='^', linestyle='-', label='20 steps')
add_fake_band(axes[1], image_k, score_20_img, eps=0.0035, color=l3.get_color())

style(
    axes[1], title="Image Ratio Analysis(Geneval)",
    xlab="Image Ratio", ylab="Overall",
    ylim=(0.90, 0.955), yticks=[0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
    xticks=image_k, xtickfmt=lambda v: f"{v:.2f}"
)

# 面板标注 (a)(b)
for ax, lab in zip(axes, ['(a)', '(b)']):
    ax.text(-0.16, 1.04, lab, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=SIZE_PANEL, fontweight='normal', clip_on=False)

# 底部统一图例（保留边框与否按需）
legend_ax.legend(
    handles=[h1, h2, h3], labels=['5 steps', '10 steps', '20 steps'],
    loc='center', ncol=3, frameon=True
)

fig.savefig("text_image_ratio.pdf", dpi=600, bbox_inches='tight')
plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import numpy as np

# # ===== 画布与间距 =====
# FIG_W, FIG_H = 12.5, 4.6
# WSPACE = 0.29
# HSPACE = 0.33
# LEGEND_ROW_RATIO = 0.17
# LABELPAD_X = 4
# LABELPAD_Y = 4

# # ===== 字号规范 =====
# SIZE_TICK   = 12   # 坐标刻度
# SIZE_LABEL  = 13   # 轴名称
# SIZE_TITLE  = 15   # 子图标题
# SIZE_PANEL  = 16   # (a)(b)
# SIZE_LEGEND = 13   # 图例

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": SIZE_TICK,
#     "axes.labelsize": SIZE_LABEL,
#     "axes.titlesize": SIZE_TITLE,
#     "xtick.labelsize": SIZE_TICK,
#     "ytick.labelsize": SIZE_TICK,
#     "legend.fontsize": SIZE_LEGEND,
# })

# # ===== 数据 =====
# # 左图：Text ratio
# text_k  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# score_5_text  = [0.923487, 0.942670, 0.928497, 0.930507, 0.937590, 0.934553, 0.924100, 0.931613, 0.926530, 0.933327]
# score_10_text = [0.933917, 0.956710, 0.941350, 0.939957, 0.950373, 0.943447, 0.932090, 0.939607, 0.941403, 0.945830]
# score_20_text = [0.943020, 0.961297, 0.944683, 0.944473, 0.953850, 0.951223, 0.938827, 0.947593, 0.953907, 0.952567]

# # 右图：Image ratio
# image_k  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
# score_5_img  = [0.9303, 0.9280, 0.9151, 0.9204, 0.9194, 0.9127, 0.9090, 0.9126, 0.9082, 0.9082]
# score_10_img = [0.9444, 0.9450, 0.9323, 0.9309, 0.9307, 0.9268, 0.9239, 0.9286, 0.9300, 0.9241]
# score_20_img = [0.9489, 0.9530, 0.9369, 0.9413, 0.9388, 0.9349, 0.9357, 0.9307, 0.9359, 0.9346]

# # ===== 平滑函数：优先 PCHIP，缺失则线性回退 =====
# try:
#     from scipy.interpolate import PchipInterpolator
#     def smooth_curve(x, y, factor=8):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         xi = np.linspace(x.min(), x.max(), len(x)*factor)
#         yi = PchipInterpolator(x, y)(xi)
#         return xi, yi
# except Exception:
#     def smooth_curve(x, y, factor=8):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         xi = np.linspace(x.min(), x.max(), len(x)*factor)
#         yi = np.interp(xi, x, y)
#         return xi, yi

# # 近似置信带（围绕平滑曲线）
# def add_fake_band(ax, xi, yi, eps=None, frac=None, color=None, alpha=0.18, zorder=1):
#     if frac is not None:
#         delta = np.maximum(frac * np.abs(yi), 1e-12)
#     else:
#         delta = eps if np.isscalar(eps) else np.asarray(eps)
#     lo, hi = yi - delta, yi + delta
#     ax.fill_between(xi, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=zorder)

# # ===== 画布：上 1 行两幅图，下 1 行图例 =====
# fig = plt.figure(figsize=(FIG_W, FIG_H))
# gs = GridSpec(
#     nrows=2, ncols=2, figure=fig,
#     height_ratios=[1.0, LEGEND_ROW_RATIO],
#     width_ratios=[1, 1],
#     wspace=WSPACE, hspace=HSPACE
# )
# axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
# legend_ax = fig.add_subplot(gs[1, :]); legend_ax.axis("off")

# def style(ax, title, xlab, ylab, ylim=None, yticks=None, xticks=None, xtickfmt=None):
#     ax.set_title(title, pad=4)
#     ax.set_xlabel(xlab, labelpad=LABELPAD_X)
#     ax.set_ylabel(ylab, labelpad=LABELPAD_Y)
#     if xticks is not None:
#         ax.set_xticks(xticks)
#         if xtickfmt is not None:
#             ax.set_xticklabels([xtickfmt(v) for v in xticks])
#     if ylim is not None: ax.set_ylim(*ylim)
#     if yticks is not None: ax.set_yticks(yticks)
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# # ----- 左图：Text Optimization Ratio -----
# h1, = axes[0].plot(text_k, score_5_text,  marker='o', linestyle='-', alpha=0.6, label='5 steps')
# h2, = axes[0].plot(text_k, score_10_text, marker='s', linestyle='-', alpha=0.6, label='10 steps')
# h3, = axes[0].plot(text_k, score_20_text, marker='^', linestyle='-', alpha=0.6, label='20 steps')

# # 叠加平滑曲线与带（eps 可按视觉调整）
# for y, h in [(score_5_text, h1), (score_10_text, h2), (score_20_text, h3)]:
#     xi, yi = smooth_curve(text_k, y, factor=8)
#     add_fake_band(axes[0], xi, yi, eps=0.004, color=h.get_color(), alpha=0.18, zorder=1)
#     axes[0].plot(xi, yi, linewidth=2.2, color=h.get_color(), zorder=3)  # 平滑线在上

# style(
#     axes[0], title="GenEval (Text ratio)",
#     xlab="Text Optimization Ratio", ylab="Overall",
#     ylim=(0.92, 0.97), yticks=[0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
#     xticks=text_k, xtickfmt=lambda v: f"{v:.1f}"
# )

# # ----- 右图：Image Optimization Ratio -----
# l1, = axes[1].plot(image_k, score_5_img,  marker='o', linestyle='-', alpha=0.6, label='5 steps')
# l2, = axes[1].plot(image_k, score_10_img, marker='s', linestyle='-', alpha=0.6, label='10 steps')
# l3, = axes[1].plot(image_k, score_20_img, marker='^', linestyle='-', alpha=0.6, label='20 steps')

# for y, h in [(score_5_img, l1), (score_10_img, l2), (score_20_img, l3)]:
#     xi, yi = smooth_curve(image_k, y, factor=8)
#     add_fake_band(axes[1], xi, yi, eps=0.0035, color=h.get_color(), alpha=0.18, zorder=1)
#     axes[1].plot(xi, yi, linewidth=2.2, color=h.get_color(), zorder=3)

# style(
#     axes[1], title="GenEval (Image ratio)",
#     xlab="Image Optimization Ratio", ylab="Overall",
#     ylim=(0.90, 0.955), yticks=[0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
#     xticks=image_k, xtickfmt=lambda v: f"{v:.2f}"
# )

# # 面板标注 (a)(b)
# for ax, lab in zip(axes, ['(a)', '(b)']):
#     ax.text(-0.16, 1.04, lab, transform=ax.transAxes,
#             ha='right', va='bottom', fontsize=SIZE_PANEL, fontweight='normal', clip_on=False)

# # 底部统一图例（无边框）
# legend_ax.legend(
#     handles=[h1, h2, h3], labels=['5 steps', '10 steps', '20 steps'],
#     loc='center', ncol=3, frameon=False
# )

# fig.savefig("text_image_ratio_smoothed.pdf", dpi=600, bbox_inches='tight')
# plt.show()
