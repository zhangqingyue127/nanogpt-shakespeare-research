import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log(path):
    text = Path(path).read_text(encoding="utf-16", errors="ignore")
    iter_x, iter_y = [], []
    eval_x, train_y, val_y = [], [], []
    for line in text.splitlines():
        m = re.search(r"iter (\d+): loss ([0-9.]+)", line)
        if m:
            iter_x.append(int(m.group(1)))
            iter_y.append(float(m.group(2)))
        m = re.search(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)", line)
        if m:
            eval_x.append(int(m.group(1)))
            train_y.append(float(m.group(2)))
            val_y.append(float(m.group(3)))
    return iter_x, iter_y, eval_x, train_y, val_y

iter_x, iter_y, eval_x, train_y, val_y = parse_log("results/logs/loss_bs64.txt")

plt.figure(figsize=(8, 5))
plt.plot(iter_x, iter_y, linewidth=1, alpha=0.45, label="Instant train loss")
plt.plot(eval_x, train_y, marker="o", linewidth=2, label="Eval train loss")
plt.plot(eval_x, val_y, marker="s", linewidth=2, linestyle="--", label="Validation loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("nanoGPT Shakespeare training curve (block_size=64)")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/loss_curve_bs64.png", dpi=300)
