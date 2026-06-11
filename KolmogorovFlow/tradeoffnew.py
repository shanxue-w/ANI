import os
import re
import matplotlib.pyplot as plt
# 移除 matplotlib.ticker，回归默认
import numpy as np

# ================= 配置区域 =================
root_dir = './'  

# 定义模型配置
model_configs = [
    ('FNO_tradeoff_pretrain', 'Pretrained FNO', '#E69F00', 'o'),  # 橙色
    ('2th', 'ANI-2', '#56B4E9', 's'),               # 天蓝色
    # ('4th', 'ANI-4', '#009E73', '^'),               # 蓝绿色
]

sizes_common = [40, 200, 400, 800, 2000, 4000]
sizes_baseline_extra = [40, 200, 400, 800, 2000, 4000]

# ================= 解析函数 (保持不变) =================

def parse_time_from_log(filepath, time_type='real'):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.compile(rf"{time_type}\s+(\d+)m(\d+\.\d+)s")
    match = pattern.search(content)
    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    else:
        return None

def parse_error_from_traj(filepath, line_index=-1):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if not lines:
            return None

        if abs(line_index) > len(lines):
            print(f"Warning: line_index {line_index} is out of range.")
            return None

        target_line = lines[line_index].strip()
        parts = target_line.split()
        
        if len(parts) >= 1:
            try:
                # 假设错误值总是该行的最后一个元素
                return float(parts[-1])
            except ValueError:
                print(f"Warning: Could not convert '{parts[-1]}' to float.")
                return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    return None

# ================= 数据加载 (保持不变) =================

results = {
    'FNO_tradeoff_pretrain': {'sizes': [], 'times': [], 'errors': []},
    '2th': {'sizes': [], 'times': [], 'errors': []},
    # '4th': {'sizes': [], 'times': [], 'errors': []}
}

for folder, label, _, _ in model_configs:
    current_sizes = sizes_baseline_extra if folder == 'FNO_tradeoff_pretrain' else sizes_common
    for size in current_sizes:
        log_path = os.path.join(root_dir, folder, f"log_{size}.txt")
        time_val = parse_time_from_log(log_path, time_type='real') 
        
        error_path = os.path.join(root_dir, folder, f"traj_error_{size}.txt")
        error_val = parse_error_from_traj(error_path, line_index=102)
        
        if time_val is not None and error_val is not None:
            results[folder]['sizes'].append(size)
            results[folder]['times'].append(time_val)
            results[folder]['errors'].append(error_val)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    # 备选方案：如果上面的样式不存在，使用经典网格样式
    plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,  # 确保字体在PDF中可编辑
    'ps.fonttype': 42
})

def create_standard_plot(data_key, title, ylabel, filename):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    for folder, label, color, marker in model_configs:
        data = results[folder]
        if not data['sizes']: continue
        
        idx = np.argsort(data['sizes'])
        x = np.array(data['sizes'])[idx]
        y = np.array(data[data_key])[idx]
        
        ax.plot(x, y, label=label, color=color, marker=marker, 
                markersize=6, linewidth=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Dataset Size (Number of Pairs)')
    ax.set_ylabel(ylabel)
    
    # 标准全封闭图框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 分别调用生成
create_standard_plot('times', 'Training Cost Analysis', 'Training Time (Seconds)', 'scaling_time_pretrain')
create_standard_plot('errors', 'Data Efficiency Analysis', r'Relative $L^2$ Error', 'scaling_error_pretrain')