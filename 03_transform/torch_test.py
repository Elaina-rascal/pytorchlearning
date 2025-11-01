import torch
import torch.nn as nn

# 测试1：检查GPU是否可用（核心）
print("GPU是否可用:", torch.cuda.is_available())
print("GPU数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前GPU名称:", torch.cuda.get_device_name(0))

# 原代码（保持不变）
a = torch.normal(0, 1, size=(3, 1))
b = torch.normal(0, 1, size=(3, 4))
print("\nCPU上的a:", a)
print("CPU上的b:", b)
print("CPU上的a*b:", a*b)

# 测试2：将数据移到GPU运行（验证GPU实际可用）
if torch.cuda.is_available():
    # 把张量移到GPU（0表示第1块GPU，多GPU可指定其他编号）
    a_gpu = a.cuda(0)
    b_gpu = b.cuda(0)
    print("\nGPU上的a_gpu:", a_gpu)
    print("GPU上的b_gpu:", b_gpu)
    print("GPU上的a_gpu*b_gpu:", a_gpu*b_gpu)
    print("a_gpu所在设备:", a_gpu.device)  # 确认是否在GPU上
else:
    print("\n未检测到可用GPU，所有计算在CPU上运行")