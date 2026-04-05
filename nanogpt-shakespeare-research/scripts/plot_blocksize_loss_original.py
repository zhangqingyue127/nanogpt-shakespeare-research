import re
import matplotlib.pyplot as plt

# 设置中文字体、负号显示及默认黑白风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray', 'darkgray'])

def parse_log(log_file):
    iter_steps = []
    iter_losses = []

    eval_steps = []
    eval_train_losses = []
    eval_val_losses = []

    with open(log_file, "r", encoding="utf-16", errors="ignore") as f:
        for line in f:
            m1 = re.search(r"iter (\d+): loss ([0-9.]+)", line)
            if m1:
                iter_steps.append(int(m1.group(1)))
                iter_losses.append(float(m1.group(2)))

            m2 = re.search(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)", line)
            if m2:
                eval_steps.append(int(m2.group(1)))
                eval_train_losses.append(float(m2.group(2)))
                eval_val_losses.append(float(m2.group(3)))

    return iter_steps, iter_losses, eval_steps, eval_train_losses, eval_val_losses

# 日志配置
logs = [
    ("BS = 32", "loss_bs32.txt"),
    ("BS = 64", "loss_bs64.txt"),
    ("BS = 128", "loss_bs128.txt"),
]

# 创建子图
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for ax, (title, logfile) in zip(axes, logs):
    iters, losses, eval_steps, eval_train, eval_val = parse_log(logfile)

    # 迭代瞬时训练损失：浅灰细实线
    ax.plot(iters, losses, alpha=0.4, label="迭代瞬时训练损失", linestyle='-', linewidth=1)
    # 评估训练损失：黑色带圆点实线
    ax.plot(eval_steps, eval_train, "o-", label="评估训练损失", linestyle='-', linewidth=2, markersize=3)
    # 验证损失：深灰带方块虚线
    ax.plot(eval_steps, eval_val, "s--", label="验证损失", linestyle='--', linewidth=2, markersize=3)

    ax.set_title(f"{title} 损失曲线", fontsize=12)
    ax.set_ylabel("损失值")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

axes[-1].set_xlabel("迭代次数")

plt.suptitle("不同block_size下的训练与验证损失曲线", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("loss_blocksize_subplots_blackwhite.png", dpi=300, bbox_inches='tight')
plt.show()