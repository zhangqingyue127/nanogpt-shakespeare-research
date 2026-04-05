import re
import matplotlib.pyplot as plt

# 设置中文字体、负号显示及默认黑白风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray', 'darkgray'])  # 黑白配色

log_file = "loss_log.txt"

iter_steps = []
train_losses = []

eval_steps = []
eval_train_losses = []
eval_val_losses = []

with open(log_file, "r", encoding="utf-16") as f:
    for line in f:
        m1 = re.search(r"iter (\d+): loss ([0-9.]+)", line)
        if m1:
            iter_steps.append(int(m1.group(1)))
            train_losses.append(float(m1.group(2)))

        m2 = re.search(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)", line)
        if m2:
            eval_steps.append(int(m2.group(1)))
            eval_train_losses.append(float(m2.group(2)))
            eval_val_losses.append(float(m2.group(3)))

plt.figure(figsize=(10, 6))
# 迭代瞬时训练损失：浅灰细实线
plt.plot(iter_steps, train_losses, label="迭代瞬时训练损失", alpha=0.6, linestyle='-', linewidth=1)
# 评估训练损失：黑色带圆点实线
plt.plot(eval_steps, eval_train_losses, "o-", label="评估训练损失", linestyle='-', linewidth=2, markersize=4)
# 验证损失：深灰带方块虚线
plt.plot(eval_steps, eval_val_losses, "s--", label="验证损失", linestyle='--', linewidth=2, markersize=4)

plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.title("nanoGPT莎士比亚文本生成模型训练与验证损失曲线")
plt.legend()
plt.grid(True, alpha=0.3)  # 网格浅化，不干扰曲线
plt.tight_layout()
plt.savefig("loss_curve_blackwhite.png", dpi=300, bbox_inches='tight')
plt.show()