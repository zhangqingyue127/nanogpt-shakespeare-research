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

pre = parse_log("results/logs/loss_preln.txt")
post = parse_log("results/logs/loss_postln.txt")

plt.figure(figsize=(8, 5))
plt.plot(pre[2], pre[4], marker="o", linewidth=2, label="Pre-LN val loss")
plt.plot(post[2], post[4], marker="s", linewidth=2, linestyle="--", label="Post-LN val loss")
plt.xlabel("Iteration")
plt.ylabel("Validation loss")
plt.title("Pre-LN vs Post-LN")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/preln_vs_postln.png", dpi=300)
