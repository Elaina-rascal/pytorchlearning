import pandas as pd
import torch
from collections import defaultdict

def LoadData(file_path,min_len=7) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    '''
    加载行人轨迹预测数据集，返回编码器输入、解码器输入、解码器输出
    编码器输入：同一时刻对应的车辆中心点及时间帧(t, x, y)
    解码器输入：行人的历史踪迹（有效长度>5）(t, x, y)
    解码器输出：与解码器输入相同
    '''
    # 读取原始xlsx文件
    data = pd.read_excel(file_path, sheet_name=None)
    df = data["sheet1"][['时间帧', 'ID', '类别', '中心坐标x', '中心坐标y']].copy()
    
    # 将类别文本转换为数字（Vehicle->0，Pedestrian->1）
    df['类别'] = df['类别'].map({'Vehicle': 0, 'Pedestrian': 1})
    
    # 分离车辆和行人数据
    vehicle_df = df[df['类别'] == 0].copy()
    pedestrian_df = df[df['类别'] == 1].copy()
    
    # 按ID分组，收集每个行人的轨迹（按时间帧排序）
    pedestrian_trajectories = defaultdict(list)
    for _, row in pedestrian_df.sort_values('时间帧').iterrows():
        pedestrian_trajectories[row['ID']].append((row['时间帧'], row['中心坐标x'], row['中心坐标y']))
    
    # 筛选出有效长度>5的行人轨迹
    valid_trajectories = []
    for traj in pedestrian_trajectories.values():
        if len(traj) > min_len:
            valid_trajectories.append(traj)
    
    # 构建解码器输入和输出（这里使用相同的轨迹，实际应用中可能需要移位）
    decoder_inputs = []
    decoder_outputs = []
    encoder_inputs = []
    
    # 为每个有效行人轨迹构建对应的输入输出
    for traj in valid_trajectories:
        # 解码器输入：行人历史轨迹 (时间帧, x, y)
        dec_in = [(t, x, y) for t, x, y in traj]
        decoder_inputs.append(dec_in)
        
        # 解码器输出：与输入相同（实际训练时可能需要做位移，如dec_in[1:] + [last_element]）
        decoder_outputs.append(dec_in)
        
        # 编码器输入：同一时刻对应的车辆中心点及时间帧 (t, x, y)
        enc_in = []
        for t, _, _ in traj:
            # 获取该时间帧所有车辆的中心点
            vehicles_at_time = vehicle_df[vehicle_df['时间帧'] == t][['中心坐标x', '中心坐标y']].values
            # 如果该时刻没有车辆，使用零向量表示（包含时间帧t）
            if len(vehicles_at_time) == 0:
                enc_in.append([t, 0.0, 0.0])  # 加入时间帧t
            else:
                # 这里使用第一个车辆的坐标，也可以修改为平均值等
                enc_in.append([t, vehicles_at_time[0][0], vehicles_at_time[0][1]])  # 加入时间帧t
        
        encoder_inputs.append(enc_in)
    
    # 转换为tensor格式（返回列表形式，因轨迹长度可能不同）
    encoder_inputs_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in encoder_inputs]
    decoder_inputs_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in decoder_inputs]
    decoder_outputs_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in decoder_outputs]
    
    return encoder_inputs_tensor, decoder_inputs_tensor, decoder_outputs_tensor

if __name__ == "__main__":
    file_path = "/pytorch/Data/data.xlsx"
    encoder_inputs, decoder_inputs, decoder_outputs = LoadData(file_path)
    print(f"样本数量: {len(encoder_inputs)}")
    if len(encoder_inputs) > 0:
        print(f"第一个样本的编码器输入形状: {encoder_inputs[0].shape}")  # 应为 (轨迹长度, 3)，3对应(t, x, y)
        print(f"第一个样本的解码器输入形状: {decoder_inputs[0].shape}")  # 应为 (轨迹长度, 3)
        print(f"第一个样本的解码器输出形状: {decoder_outputs[0].shape}")  # 应为 (轨迹长度, 3)
        print("\n第一个样本的编码器输入前3条数据:")
        print(encoder_inputs[0][:3])
        print("\n第一个样本的解码器输入前3条数据:")
        print(decoder_inputs[0][:3])