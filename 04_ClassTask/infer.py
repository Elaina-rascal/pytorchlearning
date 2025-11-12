from load_data import *
from transformer import *
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random  # 补充缺失的random导入

def infer(n: int, step_num: int, model_path: str = '/pytorch/models/task.pth'):
    """
    推理函数：输入n个行人序列和对应场景特征，预测后续step_num步并比较误差
    
    参数:
        n: 选择的行人样本数量
        step_num: 预测的步数
        model_path: 模型权重路径
    """
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    print("加载数据中...")
    file_path = "/pytorch/Data/data.xlsx"
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
    for i in range(4):
        sheet_name = 'Sheet' + str(i + 1)
        enc, dec_in, dec_out = LoadData(file_path, 7, sheet_name)
        encoder_inputs.extend(enc)
        decoder_inputs.extend(dec_in)
        decoder_outputs.extend(dec_out)
    
    # 筛选足够长的样本（至少需要step_num步的历史和预测数据）
    valid_indices = []
    for i in range(len(decoder_inputs)):
        if (len(decoder_inputs[i]) >= step_num + 10 and  # 至少10步历史+预测步
            len(encoder_inputs[i]) >= len(decoder_inputs[i])):
            valid_indices.append(i)
    
    if len(valid_indices) < n:
        raise ValueError(f"有效样本不足，需要{ n}个，实际只有{len(valid_indices)}个")
    
    # 随机选择n个样本
    selected_indices = random.sample(valid_indices, n)
    print(f"已选择{len(selected_indices)}个样本进行推理")
    
    # 2. 加载模型
    model = LightweightAttention(hidden_dim=8).to(device)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    print(f"成功加载模型，最小训练损失: {checkpoint['loss']:.4f}")
    
    # 3. 推理与比较
    loss_fn = AmplifiedResidualLoss()
    all_preds = []
    all_actuals = []
    all_errors = []
    
    with torch.no_grad():  # 关闭梯度计算
        for idx in selected_indices:
            # 获取样本数据
            enc_feat = encoder_inputs[idx].to(device)
            dec_feat = decoder_inputs[idx].to(device)
            actual = decoder_outputs[idx].to(device)
            
            # 截取历史序列（取最后能支持预测step_num步的历史）
            history_len = len(dec_feat) - step_num
            history = dec_feat[:history_len]
            actual_future = actual[history_len:history_len + step_num]
            
            # 获取对应时间的场景特征
            enc_history = enc_feat[:history_len] if len(enc_feat) >= history_len else enc_feat
            
            # 预测过程
            current_seq = history.clone()
            preds = []
            
            for _ in range(step_num):
                # 模型预测下一步
                pred_step = model(current_seq.unsqueeze(0), enc_history.unsqueeze(0))[0, -1:]
                preds.append(pred_step)
                
                # 将预测结果加入输入序列，用于下一步预测
                current_seq = torch.cat([current_seq, pred_step], dim=0)
            
            # 整理预测结果
            pred_future = torch.cat(preds, dim=0)
            
            # 计算误差
            error = loss_fn(pred_future, actual_future).item()
            l2_error = torch.norm(pred_future - actual_future, p=2).item()  # L2范数误差
            
            all_preds.append(pred_future.cpu().numpy())
            all_actuals.append(actual_future.cpu().numpy())
            all_errors.append({
                'amplified_loss': error,
                'l2_error': l2_error,
                'step_errors': torch.abs(pred_future - actual_future).mean(dim=1).cpu().numpy()
            })
            
            # 打印单样本结果
            print(f"\n样本 {idx} 预测结果:")
            print(f"  放大损失: {error:.4f}")
            print(f"  L2误差: {l2_error:.4f}")
            print(f"  平均步长误差: {np.mean(all_errors[-1]['step_errors']):.4f}")
    
    # 4. 可视化结果
    visualize_results(all_preds, all_actuals, all_errors)
    
    # 5. 输出总体统计
    avg_amplified = np.mean([e['amplified_loss'] for e in all_errors])
    avg_l2 = np.mean([e['l2_error'] for e in all_errors])
    print(f"\n总体统计:")
    print(f"  平均放大损失: {avg_amplified:.4f}")
    print(f"  平均L2误差: {avg_l2:.4f}")

def visualize_results(preds, actuals, errors):
    """可视化预测结果与实际值对比（英文显示）"""
    plt.figure(figsize=(15, 5 * len(preds)))
    
    for i in range(len(preds)):
        # 轨迹可视化
        plt.subplot(len(preds), 2, 2*i + 1)
        plt.plot(actuals[i][:, 0], actuals[i][:, 1], 'b-', label='Actual Trajectory')
        plt.plot(preds[i][:, 0], preds[i][:, 1], 'r--', label='Predicted Trajectory')
        plt.scatter(actuals[i][0, 0], actuals[i][0, 1], c='g', s=50, label='Starting Point')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Sample {i+1} Trajectory Comparison')
        plt.legend()
        plt.grid(True)
        
        # 步长误差可视化
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
        # 示例：预测5个行人的后续10步轨迹
        infer(n=5, step_num=5)
    except Exception as e:
        print(f"推理过程出错: {str(e)}")