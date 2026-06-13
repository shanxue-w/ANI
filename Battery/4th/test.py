import torch
import numpy as np
import matplotlib.pyplot as plt
from ANI4 import ANI4, load_data  # 确保能导入你的模型和数据加载器

# ================= 配置区域 =================
DATA_PATH = "../dataset/processed_battery_data_rollout.pt"
MODEL_PATH = "best_ani4_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})


# 特征索引 (对应 process_and_split_dynamic 中的 np.stack 顺序)
# [V_t, I_t, SoC_t, norm_cycle, dt_t]
V_IDX = 0       # 电压
I_IDX = 1       # 电流
SOC_IDX = 2     # SOC
CYCLE_IDX = 3   # 循环 ID (用来拆分数据)
DT_IDX = 4      # dt
# ===========================================

def recursive_prediction_single_cycle(model, cycle_x, device):
    """
    对单个 Cycle 进行递归预测
    """
    model.eval()
    predictions = []
    
    # 1. 初始化
    # 使用该 Cycle 的第一个真实点启动
    current_input = cycle_x[0].unsqueeze(0).to(device)
    
    # 为了防止 SOC 漂移，我们需要手动积分 SOC (这也是长期预测的关键)
    # 假设 Q0 = 2.0 Ah (根据 NASA B0005)
    Q_total_As = 2.0 * 3600.0 
    
    # 获取初始 SOC (从特征中读取)
    current_soc = current_input[0, SOC_IDX].item()
    
    with torch.no_grad():
        for i in range(len(cycle_x)):
            # A. 预测电压
            pred_v = model(current_input)
            predictions.append(pred_v.item())
            
            if i == len(cycle_x) - 1:
                break
            
            # B. 准备下一步输入
            # 取出真实的下一步控制量 (I, dt, CycleID 等)
            next_input_template = cycle_x[i+1].unsqueeze(0).to(device)
            
            # 获取真实的 I 和 dt 用于积分 SOC
            next_I = next_input_template[0, I_IDX].item()
            next_dt = next_input_template[0, DT_IDX].item()
            
            # --- 核心状态更新 ---
            
            # 1. 填入预测的电压 (Recursive)
            next_input_template[0, V_IDX] = pred_v
            
            # 2. 手动积分更新 SOC (这是物理约束，防止模型迷路)
            # 放电时 SOC 减小: SOC_new = SOC_old - (I * dt) / Q
            # *注意*：你的数据处理中放电电流 I_load 已经是正值
            next_soc = current_soc - (next_I * next_dt) / Q_total_As
            next_soc = max(0.0, min(1.0, next_soc)) # 截断在 0-1 之间
            
            # 填回 Tensor
            next_input_template[0, SOC_IDX] = next_soc
            
            # 更新循环变量
            current_input = next_input_template
            current_soc = next_soc
            
    return np.array(predictions)

if __name__ == "__main__":
    # 1. 加载数据
    print(f"Loading data from {DATA_PATH} ...")
    train_data, val_data, test_data, prior_params = load_data(DATA_PATH)
    
    test_x_all, test_y_all = test_data
    
    # 2. 加载模型
    print("Initializing model...")
    R_val, C_val, w_vals = prior_params
    model = ANI4(R_val, C_val, w_vals).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 3. 按 Cycle 拆分并测试
    # 获取所有唯一的 Cycle ID (注意：test_x 中的 cycle 是归一化的，或者是 float)
    # 为了准确分组，建议先转成 numpy 处理
    all_cycles_id = test_x_all[:, CYCLE_IDX].numpy()
    unique_cycles = np.unique(all_cycles_id)
    
    print(f"Found {len(unique_cycles)} unique discharge cycles in Test Set.")
    
    mse_list = []
    results = [] # 存储用于画图的数据
    
    for c_id in unique_cycles:
        # 提取当前 Cycle 的数据掩码
        mask = (all_cycles_id == c_id)
        
        # 转换为 Tensor
        cycle_x = test_x_all[mask]
        cycle_y = test_y_all[mask]
        
        if len(cycle_x) < 10: # 跳过太短的碎片
            continue
            
        # 运行递归预测
        pred_vals = recursive_prediction_single_cycle(model, cycle_x, DEVICE)
        true_vals = cycle_y.numpy().flatten()
        
        # 计算该 Cycle 的 MSE
        mse = np.mean((pred_vals - true_vals)**2)
        mse_list.append(mse)
        
        # 存储结果 (只存前中后几个典型的，防止内存爆炸)
        results.append({
            "cycle_id": c_id,
            "pred": pred_vals,
            "true": true_vals,
            "mse": mse
        })

    # 4. 统计分析
    avg_mse = np.mean(mse_list)
    print(f"\n===== Test Finished =====")
    print(f"Average Long-term MSE across {len(mse_list)} cycles: {avg_mse:.2e}")
    print(f"Worst Cycle MSE: {np.max(mse_list):.2e}")
    print(f"Best Cycle MSE: {np.min(mse_list):.2e}")

    num_plots = 1
    for j in range(len(results)):
        indices_to_plot = np.array([j], dtype=int)
        
        plt.figure(figsize=(8, 6 * num_plots))
        
        for i, idx in enumerate(indices_to_plot):
            res = results[idx]
            plt.subplot(num_plots, 1, i+1)
            
            plt.plot(res['true'], 'k-', label='Ground Truth', linewidth=2)
            plt.plot(res['pred'], 'r--', label=f'ANI-4 Pred (MSE={res["mse"]:.1e})', linewidth=1.5)

            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.xlabel("Time Step (within cycle)")
        plt.tight_layout()
        plt.savefig(f"battery_ANI4_{j}.pdf", bbox_inches='tight', format='pdf')
        plt.close()
