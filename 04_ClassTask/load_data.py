import pandas as pd
import torch
def LoadData(file_path)->tuple[torch.Tensor, torch.Tensor]:
    '''
    加载任务2的数据集
    '''
    #读取原始xlsx文件
    data = pd.read_excel(file_path, sheet_name=None)
    #将其中X1-X8提取成输入
    df = data["sheet1"][['时间帧', 'ID', '类别', '中心坐标x', '中心坐标y']].copy()
    
    # 将类别文本转换为数字（Vehicle->0，Pedestrian->1）
    df['类别'] = df['类别'].map({'Vehicle': 0, 'Pedestrian': 1})
    inputs = df.values 
    #将Y1-Y2提取成输出
    outputs = data["sheet1"][['中心坐标x', '中心坐标y']].values
    #转换成torch的tensor格式
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
    return inputs_tensor, outputs_tensor
if __name__ == "__main__":
    file_path="/pytorch/Data/data.xlsx"
    LoadData(file_path)