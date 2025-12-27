from load_data import *
from model import *
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def infer(n: int, step_num: int, model_path: str = '/pytorch/models/task.pth'):
    """
    推理函数：输入n个行人序列和对应场景特征，预测后续step_num步并比较误差
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("加载数据中...")
    file_path = "/pytorch/Data/data.xlsx"
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
    for i in range(4):
        sheet_name = 'Sheet' + str(i + 1)
        enc, dec_in, dec_out = LoadData(file_path, 7, sheet_name)
        encoder_inputs.extend(enc)
        decoder_inputs.extend(dec_in)
        decoder_outputs.extend(dec_out)
    
    valid_indices = []
    for i in range(len(decoder_inputs)):
        if (len(decoder_inputs[i]) >= step_num + 10 and 
            len(encoder_inputs[i]) >= len(decoder_inputs[i])):
            valid_indices.append(i)
    
    if len(valid_indices) < n:
        raise ValueError(f"有效样本不足，需要{ n}个，实际只有{len(valid_indices)}个")
    
    selected_indices = random.sample(valid_indices, n)
    print(f"已选择{len(selected_indices)}个样本进行推理")
    
    model = GRUEnhancer(hidden_dim=8).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"成功加载模型，最小训练损失: {checkpoint['loss']:.4f}")
    
    loss_fn = AmplifiedResidualLoss()
    all_preds = []
    all_actuals = []
    all_histories = []  # 新增：用于存储历史轨迹
    all_errors = []
    
    with torch.no_grad():
        for idx in selected_indices:
            enc_feat = encoder_inputs[idx].to(device)
            dec_feat = decoder_inputs[idx].to(device)
            actual = decoder_outputs[idx].to(device)
            
            history_len = len(dec_feat) - step_num
            history = dec_feat[:history_len]
            actual_future = actual[history_len:history_len + step_num]
            
            enc_history = enc_feat[:history_len] if len(enc_feat) >= history_len else enc_feat
            
            current_seq = history.clone()
            preds = []
            
            for _ in range(step_num):
                pred_step, weights = model(current_seq.unsqueeze(0), enc_history.unsqueeze(0))
                pred_step = pred_step.squeeze(0)[-1:]
                preds.append(pred_step)
                current_seq = torch.cat([current_seq, pred_step], dim=0)
            
            pred_future = torch.cat(preds, dim=0)
            
            error = loss_fn(pred_future, actual_future).item()
            l2_error = torch.norm(pred_future - actual_future, p=2).item()
            
            all_preds.append(pred_future.cpu().numpy())
            all_actuals.append(actual_future.cpu().numpy())
            all_histories.append(history.cpu().numpy()) # 新增：保存历史数据
            all_errors.append({
                'amplified_loss': error,
                'l2_error': l2_error,
                'step_errors': torch.abs(pred_future - actual_future).mean(dim=1).cpu().numpy()
            })
            
            print(f"\n样本 {idx} 预测结果:")
            print(f"  平均步长误差: {np.mean(all_errors[-1]['step_errors']):.4f}")
    
    visualize_results(all_preds, all_actuals, all_errors, all_histories) # 修改：传入历史数据
    
    avg_amplified = np.mean([e['amplified_loss'] for e in all_errors])
    avg_l2 = np.mean([e['l2_error'] for e in all_errors])
    print(f"\n总体统计:")
    print(f"  平均L2误差: {avg_l2:.4f}")

def visualize_results(preds, actuals, errors, histories): # 修改：增加histories参数
    """可视化预测结果与实际值对比"""
    plt.figure(figsize=(15, 5 * len(preds)))
    
    for i in range(len(preds)):
        plt.subplot(len(preds), 2, 2*i + 1)
        # 新增：绘制历史轨迹 (灰色)
        plt.plot(histories[i][:, 0], histories[i][:, 1], color='gray', linestyle='-', marker='.', label='History')
        plt.plot(actuals[i][:, 0], actuals[i][:, 1], 'b-', label='Actual Future')
        plt.plot(preds[i][:, 0], preds[i][:, 1], 'r--', label='Predicted Future')
        # 新增：标记当前点（历史的最后一点）
        plt.scatter(histories[i][-1, 0], histories[i][-1, 1], c='black', s=50, zorder=5, label='Current Pos')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Sample {i+1} Trajectory Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(len(preds), 2, 2*i + 2)
        plt.plot(errors[i]['step_errors'], 'ko-')
        plt.xlabel('Prediction Step')
        plt.ylabel('Error Value')
        plt.title(f'Sample {i+1} Step Error (Avg: {np.mean(errors[i]["step_errors"]):.4f})')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/pytorch/models/inference_results.png')
    plt.show()

if __name__ == "__main__":
    try:
        infer(n=5, step_num=5)
    except Exception as e:
        print(f"推理过程出错: {str(e)}")