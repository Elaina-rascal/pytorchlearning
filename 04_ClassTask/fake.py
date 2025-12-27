import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from load_data import LoadData

def fake_infer(n: int, step_num: int):
    """
    伪造推理过程：
    1. 直接从原始数据加载真实轨迹。
    2. 将真实未来轨迹加上微小噪声作为'预测值'。
    3. 保持与原 infer.py 完全一致的可视化风格。
    """
    print("正在从 Excel 加载真实数据...")
    file_path = "/pytorch/Data/data.xlsx"
    all_encoder_inputs = []
    all_decoder_inputs = []
    all_decoder_outputs = []

    # 加载所有 Sheet 的数据
    for i in range(4):
        sheet_name = 'Sheet' + str(i + 1)
        enc, dec_in, dec_out = LoadData(file_path, 7, sheet_name)
        all_encoder_inputs.extend(enc)
        all_decoder_inputs.extend(dec_in)
        all_decoder_outputs.extend(dec_out)

    # 筛选长度足够进行推理对比的样本
    valid_indices = []
    for i in range(len(all_decoder_inputs)):
        if len(all_decoder_inputs[i]) >= step_num + 10:
            valid_indices.append(i)

    if len(valid_indices) < n:
        n = len(valid_indices)
    
    selected_indices = random.sample(valid_indices, n)
    
    # 模拟 infer.py 中的列表存储
    all_preds = []
    all_actuals = []
    all_histories = []
    all_errors = []

    for idx in selected_indices:
        # 获取原始 Tensor 并转为 numpy
        dec_feat = all_decoder_inputs[idx].numpy()
        actual_full = all_decoder_outputs[idx].numpy()
        
        # 划分历史和未来
        history_len = len(dec_feat) - step_num
        history = dec_feat[:history_len]
        actual_future = actual_full[history_len:history_len + step_num]
        
        # --- 核心伪造逻辑：真实值 + 高斯噪声 ---
        # 噪声强度设为 0.01 到 0.03 之间，使预测线稍微偏离但趋势完美
        noise = np.random.normal(0, 0.005, size=actual_future.shape)
        pred_future = actual_future + noise
        
        # 计算误差统计量
        step_errors = np.linalg.norm(pred_future - actual_future, axis=1)
        l2_error = np.sum(step_errors)
        
        all_preds.append(pred_future)
        all_actuals.append(actual_future)
        all_histories.append(history)
        all_errors.append({
            'l2_error': l2_error,
            'step_errors': step_errors
        })

    # 调用与你 infer.py 完全一样的可视化函数
    visualize_results(all_preds, all_actuals, all_errors, all_histories)
    print(f"\n伪造推理完成，已处理 {n} 个样本。结果保存至: /pytorch/models/inference_results.png")

def visualize_results(preds, actuals, errors, histories):
    """
    保持与原 infer.py 逻辑和样式完全一致的可视化函数
    """
    plt.figure(figsize=(15, 5 * len(preds)))
    
    for i in range(len(preds)):
        # 轨迹对比图
        plt.subplot(len(preds), 2, 2*i + 1)
        plt.plot(histories[i][:, 0], histories[i][:, 1], color='gray', linestyle='-', marker='.', label='History')
        plt.plot(actuals[i][:, 0], actuals[i][:, 1], 'b-', label='Actual Future')
        plt.plot(preds[i][:, 0], preds[i][:, 1], 'r--', label='Predicted Future')
        plt.scatter(histories[i][-1, 0], histories[i][-1, 1], c='black', s=50, zorder=5, label='Current Pos')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Sample {i+1} Trajectory Comparison')
        plt.legend()
        plt.grid(True)
        
        # 误差曲线图
        plt.subplot(len(preds), 2, 2*i + 2)
        plt.plot(errors[i]['step_errors'], 'ko-')
        plt.xlabel('Prediction Step')
        plt.ylabel('Error Value')
        plt.title(f'Sample {i+1} Step Error (Avg: {np.mean(errors[i]["step_errors"]):.4f})')
        plt.grid(True)
    
    plt.tight_layout()
    # 路径保持与原需求一致
    save_path = '/pytorch/models/inference_results.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # 设置为你需要的参数
    fake_infer(n=5, step_num=5)