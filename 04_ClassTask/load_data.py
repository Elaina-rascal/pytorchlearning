import pandas as pd
import torch,random
from collections import defaultdict

def LoadData(file_path, min_len=7,sheet_name='sheet1') -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    '''
    加载行人轨迹预测数据集，返回编码器输入、解码器输入、解码器输出
    编码器输入：同一时刻对应的车辆中心点及时间帧(t, x, y)
    解码器输入：行人的历史踪迹（有效长度>5）(t, x, y)
    解码器输出：与解码器输入相同
    '''
    # 读取原始xlsx文件
    data = pd.read_excel(file_path, sheet_name=None)
    df = data[sheet_name][['时间帧', 'ID', '类别', '中心坐标x', '中心坐标y']].copy()
    
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
    #去除时间戳
    for i in range(len(encoder_inputs_tensor)):
        encoder_inputs_tensor[i]=encoder_inputs_tensor[i][:,1:]
        decoder_inputs_tensor[i]=decoder_inputs_tensor[i][:,1:]
        decoder_outputs_tensor[i]=decoder_outputs_tensor[i][:,1:]
    return encoder_inputs_tensor, decoder_inputs_tensor, decoder_outputs_tensor
def create_batch(
    encoder_inputs,
    decoder_inputs,
    decoder_outputs,
    encoder_seq_len: int,
    seq_len: int,
    batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    将数据转换为单批次格式 [batch_size, seq_len, feature_size]
    
    参数:
        encoder_inputs: 编码器输入列表（可处理任意长度，不足时会填充）
        decoder_inputs: 解码器输入列表
        decoder_outputs: 解码器输出列表
        seq_len: 每个子序列的固定长度
        batch_size: 批次大小
    
    返回:
        三个张量，形状均为 [batch_size, seq_len, feature_size]
    
    异常:
        当解码器有效样本数不足batch_size时抛出ValueError
    '''
    # 筛选出解码器长度足够的样本索引（只要求解码器序列满足长度要求）
    valid_indices = []
    for i in range(len(decoder_inputs)):
        if (len(decoder_inputs[i]) >= seq_len and 
            len(decoder_outputs[i]) >= seq_len):
            valid_indices.append(i)
    
    # 检查有效样本数是否足够
    if len(valid_indices) < batch_size:
        raise ValueError(f"解码器有效样本数不足（需要{batch_size}个，实际只有{len(valid_indices)}个）")
    
    # 随机选择batch_size个样本
    selected_indices = random.sample(valid_indices, batch_size)
    
    batch_enc = []
    batch_dec_in = []
    batch_dec_out = []
    
    for idx in selected_indices:
        # 处理编码器输入（不足长度时填充）
        enc_seq = encoder_inputs[idx]
        enc_len = len(enc_seq)
        
        if enc_len >= encoder_seq_len:
            # 长度足够时随机截取
            max_start = enc_len - encoder_seq_len
            start = random.randint(0, max_start)
            end = start + encoder_seq_len
            processed_enc = enc_seq[start:end]
        else:
            # 长度不足时填充（使用0填充）
            # 获取特征维度
            feature_size = enc_seq[0].shape[-1] if enc_len > 0 else decoder_inputs[idx][0].shape[-1]
            # 创建填充张量
            pad_length = encoder_seq_len - enc_len
            pad_tensor = torch.zeros(pad_length, feature_size, dtype=enc_seq.dtype) if enc_len > 0 else torch.zeros(pad_length, feature_size)
            processed_enc = torch.cat([enc_seq, pad_tensor], dim=0)
        
        # 处理解码器输入（保持原有逻辑）
        dec_in_seq = decoder_inputs[idx]
        dec_out_seq = decoder_outputs[idx]
        
        max_start = len(dec_in_seq) - seq_len
        start = random.randint(0, max_start)
        end = start + seq_len
        
        batch_enc.append(processed_enc)
        batch_dec_in.append(dec_in_seq[start:end])
        batch_dec_out.append(dec_out_seq[start:end])
    
    # 堆叠成 [batch_size, seq_len, feature_size] 格式的张量
    return (
        torch.stack(batch_enc),
        torch.stack(batch_dec_in),
        torch.stack(batch_dec_out)
    )


if __name__ == "__main__":
    file_path = "/pytorch/Data/data.xlsx"
    # 加载原始数据
    encoder_inputs, decoder_inputs, decoder_outputs = [],[],[]
    #增加sheet1-4的数据
    for i in range(4):
        sheet_name='Sheet'+str(i+1)
        once_encoder_inputs,once_decorder_inputs,once_decoder_outputs=LoadData(file_path,7,sheet_name)
        encoder_inputs+=once_encoder_inputs
        decoder_inputs+=once_decorder_inputs
        decoder_outputs+=once_decoder_outputs
    print(f"原始样本数量: {len(encoder_inputs)}")
    
    if len(encoder_inputs) > 0:
        # 生成单批次数据（示例参数）
        seq_len = 5  # 每个子序列的时间步长度
        batch_size = 80  # 批次大小
        
        try:
            enc_batch, dec_in_batch, dec_out_batch = create_batch(
                encoder_inputs,
                decoder_inputs,
                decoder_outputs,
                encoder_seq_len=20,
                seq_len=seq_len,
                batch_size=batch_size
            )
            
            # 输出形状：[batch_size, seq_len, feature_size]
            print(f"\n编码器输入批次形状: {enc_batch.shape}")    # [3, 5, 3]
            print(f"解码器输入批次形状: {dec_in_batch.shape}")  # [3, 5, 3]
            print(f"解码器输出批次形状: {dec_out_batch.shape}")  # [3, 5, 3]
            
            # 打印第一个样本的前3个时间步
            print("\n第一个样本的编码器输入（前3步）:")
            print(enc_batch[0, :3])
            
        except ValueError as e:
            print(f"错误: {e}")