# pytorchlearning

## 04_ClassTask 使用说明

本目录包含行人轨迹预测任务的训练和推理代码。

### 环境准备

本项目提供了 `docker-compose.yml` 文件（位于 `.devcontainer` 目录下），可以直接使用 Docker Compose 启动带有 NVIDIA Runtime 支持的开发环境。[nvidia runtime安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **进入配置目录**

   ```bash
   cd .devcontainer
   ```

2. **启动 GPU 容器**

   ```bash
   docker-compose up 
   ```

3. **进入容器**

   ```bash
   docker exec -it pytorch-gpu-container /bin/bash
   ```

### 目录结构

- `train.py`: 模型训练脚本
- `infer.py`: 模型推理与评估脚本
- `model.py`: 模型定义 (GRUEnhancer)
- `load_data.py`: 数据加载工具
- `visualAndSave.py`: 训练过程可视化与模型保存工具

### 使用方法

#### 1. 训练模型

运行以下命令开始训练模型：

```bash
python 04_ClassTask/train.py
```

- 训练过程中，模型会自动保存到 `/pytorch/models/task.pth`。
- 损失曲线图会保存到 `/pytorch/models/loss_curve.png`。
- 训练数据来自 `/pytorch/Data/data.xlsx`。

#### 2. 模型推理

使用训练好的模型进行推理和评估：

```bash
python 04_ClassTask/infer.py
```

- 该脚本会随机选择 5 个样本进行推理，预测未来 5 步的轨迹。
- 推理结果（轨迹对比图和误差分析）将保存为 `/pytorch/models/inference_results.png`。
