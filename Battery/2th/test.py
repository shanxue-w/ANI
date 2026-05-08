import torch
import numpy as np
import matplotlib.pyplot as plt
from ANI2 import ANI2, load_data  # 确保能导入你的模型和数据加载器

# ================= 配置区域 =================
DATA_PATH = "../dataset/processed_battery_data_rollout.pt"
MODEL_PATH = "best_ani2_model.pth"
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
    
    Q_total_As = 2.0 * 3600.0 
    
    current_soc = current_input[0, SOC_IDX].item()
    
    with torch.no_grad():
        for i in range(len(cycle_x)):
            pred_v = model(current_input)
            predictions.append(pred_v.item())
            
            if i == len(cycle_x) - 1:
                break
            
            next_input_template = cycle_x[i+1].unsqueeze(0).to(device)
            
            next_I = next_input_template[0, I_IDX].item()
            next_dt = next_input_template[0, DT_IDX].item()
            
            next_input_template[0, V_IDX] = pred_v
            
            next_soc = current_soc - (next_I * next_dt) / Q_total_As
            next_soc = max(0.0, min(1.0, next_soc)) # 截断在 0-1 之间
            
            next_input_template[0, SOC_IDX] = next_soc
            
            current_input = next_input_template
            current_soc = next_soc
            
    return np.array(predictions)

if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH} ...")
    train_data, val_data, test_data, prior_params = load_data(DATA_PATH)
    
    test_x_all, test_y_all = test_data
    
    print("Initializing model...")
    R_val, C_val, w_vals = prior_params
    model = ANI2(R_val, C_val, w_vals).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    all_cycles_id = test_x_all[:, CYCLE_IDX].numpy()
    unique_cycles = np.unique(all_cycles_id)
    
    print(f"Found {len(unique_cycles)} unique discharge cycles in Test Set.")
    
    mse_list = []
    results = []
    
    for c_id in unique_cycles:
        mask = (all_cycles_id == c_id)
        
        cycle_x = test_x_all[mask]
        cycle_y = test_y_all[mask]
        
        if len(cycle_x) < 10: 
            continue
            
        pred_vals = recursive_prediction_single_cycle(model, cycle_x, DEVICE)
        true_vals = cycle_y.numpy().flatten()
        
        mse = np.mean((pred_vals - true_vals)**2)
        mse_list.append(mse)
        
        results.append({
            "cycle_id": c_id,
            "pred": pred_vals,
            "true": true_vals,
            "mse": mse
        })

    avg_mse = np.mean(mse_list)
    print(f"\n===== Test Finished =====")
    print(f"Average Long-term MSE across {len(mse_list)} cycles: {avg_mse:.2e}")
    print(f"Worst Cycle MSE: {np.max(mse_list):.2e}")
    print(f"Best Cycle MSE: {np.min(mse_list):.2e}")

    num_plots = 1
    # indices_to_plot = np.linspace(0, len(results)-1, num_plots, dtype=int)
    for j in range(len(results)):
        indices_to_plot = np.array([j], dtype=int)
        
        plt.figure(figsize=(8, 6 * num_plots))
        
        for i, idx in enumerate(indices_to_plot):
            res = results[idx]
            plt.subplot(num_plots, 1, i+1)
            
            plt.plot(res['true'], 'k-', label='Ground Truth', linewidth=2)
            plt.plot(res['pred'], 'r--', label=f'ANI-2 Pred (MSE={res["mse"]:.1e})', linewidth=1.5)
            
            # plt.title(f"Cycle ID (norm): {res['cycle_id']:.4f}")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.xlabel("Time Step (within cycle)")
        plt.tight_layout()
        # plt.savefig("multicycle_test_result.png")
        # print("Result saved to multicycle_test_result.png")
        plt.savefig(f"battery_ANI2_{j}.pdf", bbox_inches='tight', format='pdf')
        plt.close()