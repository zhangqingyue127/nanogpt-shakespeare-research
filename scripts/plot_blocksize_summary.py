import csv
import matplotlib.pyplot as plt

rows = []
with open("results/block_size_summary.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

block_sizes = [int(r["block_size"]) for r in rows]
val_losses = [float(r["final_val_loss"]) for r in rows]
times = [float(r["avg_iter_time_ms"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(block_sizes, val_losses, marker="o", linewidth=2)
ax1.set_xlabel("Block size")
ax1.set_ylabel("Final validation loss")
ax1.set_title("Block size trade-off")

ax2 = ax1.twinx()
ax2.plot(block_sizes, times, marker="s", linewidth=2, linestyle="--")
ax2.set_ylabel("Average iteration time (ms)")

plt.tight_layout()
plt.savefig("results/figures/block_size_tradeoff.png", dpi=300)
