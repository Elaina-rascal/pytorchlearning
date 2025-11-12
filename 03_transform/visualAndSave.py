# visual.py
import matplotlib.pyplot as plt
import os,torch

def plot_loss(losses, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()
class SaveAndVisual:
    """模型管理类，负责模型保存和训练可视化"""
    
    def __init__(self, model_dir='models', loss_img_path='loss_curve.png'):
        self.model_dir = model_dir
        self.loss_img_path = loss_img_path
        self.epoch_losses = []  # 存储每个epoch的损失
        self.epoch_indices = []  # 存储epoch索引
        self._init_visualization()
        self._init_model_dir()

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

    def load_model(self, model, optimizer, device):
        """加载已保存的模型"""
        model_path = os.path.join(self.model_dir, 'best_transformer.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"发现现有模型，其损失为: {checkpoint['loss']:.4f}")
            return checkpoint['loss']
        return float('inf')

    def save_model(self, model, optimizer, epoch, loss):
        """保存模型检查点"""
        model_path = os.path.join(self.model_dir, 'best_transformer.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)
        print(f"  保存最佳模型（损失：{loss:.4f})")

    def update_visualization(self, epoch, loss):
        """更新训练损失可视化"""
        self.epoch_losses.append(loss)
        self.epoch_indices.append(epoch)
        
        # 更新图像数据
        self.line.set_data(self.epoch_indices, self.epoch_losses)
        self.ax.relim()  # 重新计算坐标轴范围
        self.ax.autoscale_view()  # 自动调整视图
        plt.draw()
        plt.pause(0.01)

    def finalize_visualization(self):
        """训练结束后保存并显示最终图像"""
        plt.ioff()  # 关闭交互模式
        self.ax.set_title("Training Loss (Final)")
        plt.savefig(self.loss_img_path)
        plt.show()
