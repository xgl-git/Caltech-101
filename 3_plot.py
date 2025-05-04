import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 可选：设置日志目录，分别对应 finetune 和 scratch 模型
tags = ['finetune', 'scratch']
log_dirs = {
    tag: os.path.join('runs', f'caltech101_{tag}') for tag in tags
}

# 读取 TensorBoard 数据
def load_tensorboard_data(log_dir, scalar_tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    return ea.Scalars(scalar_tag)

# 绘图函数
def plot_scalar(tag_name, scalar_name, ylabel):
    plt.figure(figsize=(8, 5))
    for tag in tags:
        path = log_dirs[tag]
        data = load_tensorboard_data(path, f"{tag}/{scalar_name}")
        steps = [d.step for d in data]
        values = [d.value for d in data]
        plt.plot(steps, values, label=tag)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{scalar_name.replace("_", " ").title()} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{scalar_name}_{tag_name}.png')
    plt.show()

# 绘制对比图
plot_scalar('loss', 'train_loss', 'Training Loss')
plot_scalar('loss', 'val_loss', 'Validation Loss')
plot_scalar('acc', 'val_acc', 'Validation Accuracy')