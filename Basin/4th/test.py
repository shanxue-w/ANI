import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# 从 ANI2.py 导入模型类和配置
from ANI4 import ANI_4th_Hydro, BASIN_ID, DATA_FILE, HIDDEN_DIM, SEQ_LEN, device

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

# 模型权重文件
MODEL_PATH = f"ANI2_Best_{BASIN_ID}.pth"

def calculate_metrics(obs, sim):
    obs = obs.flatten()
    sim = sim.flatten()
    
    # NSE
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    
    # KGE
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-8)
    beta = np.mean(sim) / (np.mean(obs) + 1e-8)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    # PBIAS
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)
    
    return {"NSE": nse, "KGE": kge, "PBIAS": pbias}

def run_test():
    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
    full_data = torch.load(DATA_FILE, weights_only=False)
    
    test_X_raw = full_data['test']['X_raw'].to(device)   # [N, 6]
    test_X_norm = full_data['test']['X_norm'].to(device) # [N, 6]
    test_y_raw = full_data['test']['y_raw'].to(device)             # [N, 1]
    
    model = ANI_4th_Hydro(
        ode_prior=full_data['ode_prior'],
        stats=full_data['stats'],
        input_dim=int(full_data['train']['X_norm'].shape[-1]),
        hidden_dim=HIDDEN_DIM
    ).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Successfully loaded model: {MODEL_PATH}")

    history_norm = test_X_norm[:SEQ_LEN].unsqueeze(0) 
    q_start = test_y_raw[SEQ_LEN-1 : SEQ_LEN].to(device) 
    x_future_raw = test_X_raw[SEQ_LEN-1 : ].unsqueeze(0)  
    x_future_norm = test_X_norm[SEQ_LEN-1 : ].unsqueeze(0) 

    pred_steps = len(test_X_raw) - SEQ_LEN

    print(f"Starting rollout for {pred_steps} days...")

    with torch.no_grad():
        y_pred, _ = model(history_norm, x_future_raw, x_future_norm, q_start, pred_steps)
    
    preds = y_pred.squeeze().cpu().numpy()
    obs = test_y_raw[SEQ_LEN:].cpu().squeeze().numpy()

    # 5. 计算指标
    m = calculate_metrics(obs, preds)
    print("\n" + "="*30)
    print(f"Test Results (Basin {BASIN_ID}):")
    print(f"NSE:   {m['NSE']:.4f}")
    print(f"KGE:   {m['KGE']:.4f}")
    print(f"PBIAS: {m['PBIAS']:.2f}%")
    print("="*30)

    # 6. 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(obs, label='Observed', color='black', alpha=0.7, lw=1.5)
    plt.plot(preds, label='Predicted (ANI4)', color='red', alpha=0.8, lw=1.2, ls='--')
    # plt.title(f"Long-term Simulation - Basin {BASIN_ID}")
    plt.xlabel("Days")
    plt.ylabel("Streamflow (mm/day)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注指标
    res_text = f"NSE: {m['NSE']:.3f}\nKGE: {m['KGE']:.3f}\nPBIAS: {m['PBIAS']:.1f}%"
    plt.text(0.02, 0.95, res_text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"rollout_result_{BASIN_ID}_prior.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print(f"Plot saved to rollout_result_{BASIN_ID}_prior.pdf")
    plt.show()

# # 设置 2行2列 的画板
#     fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
#     # 子图1: 时间序列 (Time Series)
#     ax1 = axes[0, 0]
#     ax1.plot(obs, label='Observed', color='black', alpha=0.6, lw=1)
#     ax1.plot(preds, label='Predicted', color='red', alpha=0.7, lw=1, ls='--')
#     ax1.set_title(f"Hydrograph - Basin {BASIN_ID}")
#     ax1.set_ylabel("Flow")
#     ax1.legend()
#     ax1.text(0.02, 0.95, f"NSE: {m['NSE']:.3f}\nKGE: {m['KGE']:.3f}", 
#              transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

#     # 子图2: 散点图 (Scatter Plot)
#     ax2 = axes[0, 1]
#     ax2.scatter(obs, preds, alpha=0.3, s=10, c='blue')
#     max_val = max(obs.max(), preds.max())
#     ax2.plot([0, max_val], [0, max_val], 'k--', lw=1.5) # 1:1 线
#     ax2.set_title("Scatter Plot (Obs vs Sim)")
#     ax2.set_xlabel("Observed")
#     ax2.set_ylabel("Predicted")

#     # 子图3: 流量历时曲线 (FDC) - Log Scale
#     ax3 = axes[1, 0]
#     sort_obs = np.sort(obs)[::-1] # 降序
#     sort_sim = np.sort(preds)[::-1]
#     exceed_prob = np.arange(1, len(obs)+1) / len(obs)
    
#     ax3.plot(exceed_prob, sort_obs, label='Obs', color='black', lw=1.5)
#     ax3.plot(exceed_prob, sort_sim, label='Sim', color='red', ls='--', lw=1.5)
#     ax3.set_yscale('log') # 对数坐标看低流更清楚
#     ax3.set_title("Flow Duration Curve (Log Scale)")
#     ax3.set_xlabel("Exceedance Probability")
#     ax3.set_ylabel("Flow (Log)")
#     ax3.legend()

#     # 子图4: 局部放大 (Zoom in) - 比如看某一个具体的洪水事件
#     # 这里取前100天或者最大的洪水区间
#     ax4 = axes[1, 1]
#     zoom_slice = slice(100, 200) # 示例：看第100到200天
#     ax4.plot(obs[zoom_slice], 'k-', label='Obs')
#     ax4.plot(preds[zoom_slice], 'r--', label='Sim')
#     ax4.set_title("Zoomed In Window (100-200 days)")
    
#     plt.tight_layout()
#     plt.savefig(f"comprehensive_analysis_{BASIN_ID}.png", dpi=300)
#     plt.show()

if __name__ == "__main__":
    run_test()


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm

# # 假设这些是你之前的配置
# from ANI4 import ANI_4th_Hydro, BASIN_ID, DATA_FILE, HIDDEN_DIM, SEQ_LEN, device

# # ================= 配置 =================
# MODEL_PATH = f"ANI2_Best_{BASIN_ID}.pth"  # 确保加载的是优化后的模型
# TEST_WINDOW_SIZE = 30   # 每个窗口预测多少天 (建议比训练时的30天稍长，测试泛化)
# TEST_STRIDE = 10        # 窗口滑动的步长 (每隔10天测一次)
# METRIC_TYPE = "NSE"     # 主要关注的指标

# def calculate_single_window_metrics(obs, sim):
#     """计算单个窗口的指标"""
#     if len(obs) < 2: return -np.inf # 避免极其短的数据
    
#     # 避免分母为0 (如果观测值是常数)
#     obs_var = np.var(obs)
#     if obs_var < 1e-6:
#         return np.nan 

#     nse = 1 - np.sum((obs - sim)**2) / (np.sum((obs - np.mean(obs))**2) + 1e-8)
#     return nse

# def run_sliding_test():
#     # 1. 加载数据
#     if not os.path.exists(DATA_FILE):
#         print(f"Error: {DATA_FILE} not found.")
#         return
#     full_data = torch.load(DATA_FILE, weights_only=False)
    
#     # 获取测试集数据
#     test_X_raw = full_data['test']['X_raw'].to(device)
#     test_X_norm = full_data['test']['X_norm'].to(device)
#     test_y_raw = full_data['test']['y_raw'].to(device)
    
#     # 2. 加载模型
#     # 注意：这里需要实例化你训练时用到的那个模型类 (比如 ANI_2th_Hydro 或 ANI_4th_Hydro)
#     # 假设你使用的是 ANI_2th_Hydro
#     input_dim = int(full_data['train']['X_norm'].shape[-1])
    
#     model = ANI_4th_Hydro(
#         ode_prior=full_data['ode_prior'],
#         stats=full_data['stats'],
#         input_dim=input_dim,
#         hidden_dim=16 # 注意：一定要和训练时的 hidden_dim 一致！之前代码改成了32
#     ).to(device)
    
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: {MODEL_PATH} not found.")
#         return
        
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#     print(f"Loaded model from {MODEL_PATH}")

#     # 3. 滑动窗口测试
#     total_len = len(test_X_raw)
#     num_windows = (total_len - SEQ_LEN - TEST_WINDOW_SIZE) // TEST_STRIDE + 1
    
#     print(f"Total Test Days: {total_len}, Windows: {num_windows}")
#     print(f"Window Size: {TEST_WINDOW_SIZE} days, Stride: {TEST_STRIDE} days")

#     nse_list = []
#     # 用于记录每一步的误差 (Time-step wise error)
#     # shape: [TEST_WINDOW_SIZE], 存储每一步的累计平方误差和计数
#     mse_per_step = np.zeros(TEST_WINDOW_SIZE)
#     count_per_step = np.zeros(TEST_WINDOW_SIZE)
    
#     all_obs_windows = []
#     all_pred_windows = []

#     with torch.no_grad():
#         for i in tqdm(range(0, total_len - SEQ_LEN - TEST_WINDOW_SIZE, TEST_STRIDE)):
#             # 准备当前窗口的数据
#             # History: [i : i+SEQ_LEN]
#             hist_norm = test_X_norm[i : i+SEQ_LEN].unsqueeze(0)
#             q_start = test_y_raw[i+SEQ_LEN-1].unsqueeze(0) # Scalar or [1,1]
            
#             # Future: [i+SEQ_LEN-1 : i+SEQ_LEN+TEST_WINDOW_SIZE]
#             # 注意: x_future 需要包含 t=0 (即 q_start 对应的时间) 的 forcing，
#             # 但模型内部通常取 t+1。这里直接取够长度。
#             f_raw = test_X_raw[i+SEQ_LEN-1 : i+SEQ_LEN+TEST_WINDOW_SIZE].unsqueeze(0)
#             f_norm = test_X_norm[i+SEQ_LEN-1 : i+SEQ_LEN+TEST_WINDOW_SIZE].unsqueeze(0)
            
#             target = test_y_raw[i+SEQ_LEN : i+SEQ_LEN+TEST_WINDOW_SIZE].cpu().numpy().flatten()
            
#             # 预测
#             y_pred, _ = model(hist_norm, f_raw, f_norm, q_start, TEST_WINDOW_SIZE)
#             pred = y_pred.squeeze().cpu().numpy()
            
#             # 记录指标
#             nse = calculate_single_window_metrics(target, pred)
#             if not np.isnan(nse):
#                 nse_list.append(nse)
            
#             # 记录用于 Horizon Analysis 的误差
#             sq_diff = (pred - target) ** 2
#             mse_per_step += sq_diff
#             count_per_step += 1
            
#             # 随机抽样几个窗口用于画图展示 (比如存前3个，中间3个)
#             if i % (num_windows // 5) == 0:
#                 all_obs_windows.append(target)
#                 all_pred_windows.append(pred)

#     # 4. 结果可视化
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig = plt.figure(figsize=(18, 10))
#     gs = fig.add_gridspec(2, 2)

#     # --- 图1: NSE 分布箱线图 ---
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.boxplot(nse_list, vert=False, patch_artist=True, 
#                 boxprops=dict(facecolor='lightblue', color='blue'),
#                 medianprops=dict(color='red'))
#     ax1.set_title(f'NSE Distribution over {len(nse_list)} Windows')
#     ax1.set_xlabel('NSE')
#     ax1.axvline(np.mean(nse_list), color='green', linestyle='--', label=f'Mean: {np.mean(nse_list):.3f}')
#     ax1.legend()

#     # --- 图2: 误差随时间步长衰减 (Horizon Analysis) ---
#     ax2 = fig.add_subplot(gs[0, 1])
#     rmse_per_step = np.sqrt(mse_per_step / count_per_step)
#     steps = np.arange(1, TEST_WINDOW_SIZE + 1)
#     ax2.plot(steps, rmse_per_step, marker='o', markersize=4, color='darkorange', linewidth=2)
#     ax2.set_title('Forecast Error Growth (RMSE vs. Lead Time)')
#     ax2.set_xlabel('Lead Time (Days)')
#     ax2.set_ylabel('RMSE (mm/day)')
#     ax2.grid(True, alpha=0.3)

#     # --- 图3: 几个典型窗口的对比展示 ---
#     ax3 = fig.add_subplot(gs[1, :])
#     for idx, (o, p) in enumerate(zip(all_obs_windows[:5], all_pred_windows[:5])):
#         # 为了不重叠，把它们拼在一起或者只画一部分
#         # 这里简单画在一起，加上偏移量
#         offset = idx * (TEST_WINDOW_SIZE + 10)
#         x_range = np.arange(offset, offset + len(o))
#         ax3.plot(x_range, o, color='black', alpha=0.6, label='Obs' if idx==0 else "")
#         ax3.plot(x_range, p, color='red', alpha=0.8, linestyle='--', label='Pred' if idx==0 else "")
#         # 画竖线分隔
#         ax3.axvline(offset + len(o), color='gray', linestyle=':', alpha=0.3)
    
#     ax3.set_title('Sample Rollout Windows (Concatenated)')
#     ax3.set_xlabel('Concatenated Time Steps')
#     ax3.legend()

#     plt.tight_layout()
#     plt.savefig(f"sliding_window_test_{BASIN_ID}.png", dpi=300)
#     print(f"\nSaved analysis to sliding_window_test_{BASIN_ID}.png")
    
#     # 打印统计摘要
#     print("\n" + "="*40)
#     print(f"Sliding Window Test Summary ({len(nse_list)} samples)")
#     print("="*40)
#     print(f"Mean NSE:   {np.mean(nse_list):.4f}")
#     print(f"Median NSE: {np.median(nse_list):.4f}")
#     print(f"Std Dev:    {np.std(nse_list):.4f}")
#     print(f"Min NSE:    {np.min(nse_list):.4f}")
#     print(f"Max NSE:    {np.max(nse_list):.4f}")
#     print("="*40)

# if __name__ == "__main__":
#     run_sliding_test()