# train a miniature character-level shakespeare model
# (course experiment version, CPU-friendly)

# ===== system =====
device = 'cpu'
compile = False

# ===== logging =====
out_dir = 'out-shakespeare-char'
eval_interval = 250
eval_iters = 20
log_interval = 10
always_save_checkpoint = False
wandb_log = False

# ===== dataset =====
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64

# ===== model (small GPT) =====
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# ===== optimizer =====
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
