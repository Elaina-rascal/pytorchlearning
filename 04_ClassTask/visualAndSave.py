# visual.py
import matplotlib.pyplot as plt
import os,torch
class SaveAndVisual:
    """模型管理类，负责模型保存和训练可视化"""
    
    def __init__(self, model_dir='models', loss_img_path='loss_curve.png'):
        self.model_dir = model_dir
        self.loss_img_path = loss_img_path
        self.epoch_losses = []  # 存储每个epoch的损失
        self.epoch_indices = []  # 存储epoch索引
        self._init_visualization()
        # self._init_model_dir()
        self.loop_count=0
    def _init_model_dir(self):
        """初始化模型保存目录"""
        os.makedirs(self.model_dir, exist_ok=True)

    def _init_visualization(self):
        """初始化可视化环境"""
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss (per Epoch)")
        self.line, = self.ax.plot([], [], label="Epoch Loss")
        self.ax.legend()
        self.min_loss=float('inf')
    def loadModel(self, model:torch.nn.Module, optimizer, device):
        """加载已保存的模型"""
        model_path = os.path.join(self.model_dir)
        self.model=model
        self.optimizer=optimizer
        #检查末尾是否为.pth
    # 检查路径是否以.pth结尾
        if not model_path.endswith('.pth'):
            raise ValueError(f"模型路径'{model_path}'应以.pth结尾")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"模型文件'{model_path}'不存在，将从头训练")
            return None
        # 检查路径是否为文件（不是文件夹）
        # if not os.path.isfile(model_path):
        #     raise IsADirectoryError(f"路径'{model_path}'是文件夹，不是模型文件（请传入完整的文件路径）")
        
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"成功加载模型'{model_path}'，其损失为: {checkpoint['loss']:.4f}")
        self.min_loss = checkpoint['loss']
        return checkpoint['loss']

    def saveModel(self, model, optimizer, epoch, loss):
        """保存模型检查点"""
        model_path = os.path.join(self.model_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)
        # print(f"  保存最佳模型（损失：{loss:.4f})")

    def updateVisualization(self, epoch, loss):
        """更新训练损失可视化"""
        self.epoch_losses.append(loss)
        self.epoch_indices.append(epoch)
        self.loop_count+=1
        if(self.loop_count%10==0):
            self.loop_count=0
            print(f"  训练损失（第{epoch+1}轮）：{loss:.4f}")
        # 更新图像数据
        self.line.set_data(self.epoch_indices, self.epoch_losses)
        self.ax.relim()  # 重新计算坐标轴范围
        self.ax.autoscale_view()  # 自动调整视图
        if(loss<self.min_loss ):
            self.min_loss=loss
            self.saveModel(self.model,self.optimizer,epoch,loss)
        plt.draw()
        plt.pause(0.01)

    def finalizeVisualization(self):
        """训练结束后保存并显示最终图像"""
        plt.ioff()  # 关闭交互模式
        self.ax.set_title("Training Loss (Final)")
        plt.savefig(self.loss_img_path)
        plt.show()
if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r-', label='test')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('GUI')
    plt.legend()
    plt.grid(True)
    plt.show()  # 弹出窗口显示测试图