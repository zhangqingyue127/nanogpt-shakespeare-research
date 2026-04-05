import re
import matplotlib.pyplot as plt

# 设置中文字体、负号显示及默认黑白风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体，Linux替换为['WenQuanYi Zen Hei']，Mac替换为['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray', 'darkgray'])

# ================== 完整日志字符串（移除省略号，保留真实数据）==================
loss_preln_text = """
step 0: train loss 4.1676, val loss 4.1649
iter 0: loss 4.1828
iter 100: loss 2.6635
iter 200: loss 2.5039
step 250: train loss 2.4293, val loss 2.4447
iter 300: loss 2.3350
iter 400: loss 2.3405
step 500: train loss 2.2732, val loss 2.3141
iter 500: loss 2.3827
iter 600: loss 2.1762
iter 700: loss 2.0947
step 750: train loss 2.1338, val loss 2.1905
iter 800: loss 2.2092
iter 900: loss 2.0011
step 1000: train loss 1.9714, val loss 2.0528
iter 1000: loss 2.0455
iter 1100: loss 1.7907
iter 1200: loss 1.9533
step 1250: train loss 1.8756, val loss 2.0089
iter 1300: loss 1.9599
iter 1400: loss 1.8251
step 1500: train loss 1.8557, val loss 1.9225
iter 1500: loss 1.8624
iter 1600: loss 1.7808
iter 1700: loss 1.7163
step 1750: train loss 1.7790, val loss 1.8921
iter 1800: loss 1.8415
iter 1900: loss 1.6849
step 2000: train loss 1.7648, val loss 1.8857
iter 2000: loss 1.6958
"""

loss_postln_text = """
step 0: train loss 4.2005, val loss 4.1940
iter 0: loss 4.2126
iter 100: loss 2.7977
iter 200: loss 2.5156
step 250: train loss 2.4621, val loss 2.4813
iter 300: loss 2.3801
iter 400: loss 2.3572
step 500: train loss 2.2789, val loss 2.3150
iter 500: loss 2.3838
iter 600: loss 2.1586
iter 700: loss 2.0958
step 750: train loss 2.1178, val loss 2.1614
iter 800: loss 2.1841
iter 900: loss 2.0130
step 1000: train loss 1.9795, val loss 2.0462
iter 1000: loss 2.0453
iter 1100: loss 1.8027
iter 1200: loss 1.9771
step 1250: train loss 1.8954, val loss 2.0181
iter 1300: loss 1.9703
iter 1400: loss 1.8153
step 1500: train loss 1.8786, val loss 1.9423
iter 1500: loss 1.8959
iter 1600: loss 1.8304
iter 1700: loss 1.7606
step 1750: train loss 1.8185, val loss 1.9222
iter 1800: loss 1.8582
iter 1900: loss 1.7479
step 2000: train loss 1.8093, val loss 1.9157
iter 2000: loss 1.7252
"""

# ================== 解析函数 ==================
def parse_log_from_text(text):
    iters, losses = [], []
    eval_steps, train_eval, val_eval = [], [], []
    for line in text.splitlines():
        # 匹配迭代瞬时损失
        m1 = re.search(r"iter (\d+): loss ([0-9.]+)", line.strip())
        if m1:
            iters.append(int(m1.group(1)))
            losses.append(float(m1.group(2)))
        # 匹配step评估损失
        m2 = re.search(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)", line.strip())
        if m2:
            eval_steps.append(int(m2.group(1)))
            train_eval.append(float(m2.group(2)))
            val_eval.append(float(m2.group(3)))
    return iters, losses, eval_steps, train_eval, val_eval

# ================== 解析数据 ==================
pre_i, pre_l, pre_es, pre_tr, pre_val = parse_log_from_text(loss_preln_text)
post_i, post_l, post_es, post_tr, post_val = parse_log_from_text(loss_postln_text)

# ================== 创建黑白风格子图 ==================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Pre-LN子图（黑白区分：实线+不同标记）
axes[0].plot(pre_i, pre_l, alpha=0.4, label="迭代瞬时训练损失", linestyle='-', linewidth=1)
axes[0].plot(pre_es, pre_tr, "o-", label="评估训练损失", linestyle='-', linewidth=2, markersize=3)
axes[0].plot(pre_es, pre_val, "s--", label="验证损失", linestyle='--', linewidth=2, markersize=3)
axes[0].set_title("Pre-LN结构损失曲线", fontsize=12)
axes[0].set_ylabel("损失值")
axes[0].grid(True, alpha=0.3)  # 浅灰网格不干扰曲线
axes[0].legend(fontsize=10)

# Post-LN子图（保持相同黑白样式，确保对比一致性）
axes[1].plot(post_i, post_l, alpha=0.4, label="迭代瞬时训练损失", linestyle='-', linewidth=1)
axes[1].plot(post_es, post_tr, "o-", label="评估训练损失", linestyle='-', linewidth=2, markersize=3)
axes[1].plot(post_es, post_val, "s--", label="验证损失", linestyle='--', linewidth=2, markersize=3)
axes[1].set_title("Post-LN结构损失曲线", fontsize=12)
axes[1].set_xlabel("迭代次数")
axes[1].set_ylabel("损失值")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

# 总标题
plt.suptitle("Transformer Pre-LN与Post-LN结构损失对比", fontsize=14, y=1.02)
plt.tight_layout()
# 保存黑白图片（dpi=300保证清晰度，适配论文印刷）
plt.savefig("preln_vs_postln_loss_blackwhite.png", dpi=300, bbox_inches='tight')
plt.show()