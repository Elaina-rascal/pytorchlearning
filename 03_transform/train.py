from transformer import *
from load_data import *
import torch,os
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
def train():
    num_hiddens, num_layers, dropout, batch_size = 32, 2, 0.1, 64
    lr, num_epochs, device = 0.005, 200, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    data_path = "/home/Elaina/pytorch/Data/cmn.txt"
    src_lang, tgt_lang = 'en', 'zh'
    src_sents, tgt_sents = read_translation_data(data_path)
    print(f"读取到 {len(src_sents)} 条平行语料")
    
    # 词元化
    src_tokens = tokenize(src_sents, src_lang)
    tgt_tokens = tokenize(tgt_sents, tgt_lang)
    
    # 构建词表
    reserved = ['<pad>', '<bos>', '<eos>', '<unk>']  # 特殊符号：填充、句首、句尾、未知
    src_vocab = Vocab(src_tokens, min_freq=1, reserved_tokens=reserved)  # 测试时将min_freq设为1
    tgt_vocab = Vocab(tgt_tokens, min_freq=1, reserved_tokens=reserved)
    print(f"源语言词表大小：{len(src_vocab)}，目标语言词表大小：{len(tgt_vocab)}")
    dataset = TranslationDataset(
        src_tokens, tgt_tokens, 
        src_vocab, tgt_vocab
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,  # 打乱数据顺序
        drop_last=True,  # 丢弃最后一个不完整的批次
        collate_fn=custom_collate_fn  # 使用自定义的批量处理函数
    )
    # 这里需要to(device)把模型和数据都放到同一个设备上（CPU或GPU）
    encoder=Encoder(vocab_size=len(src_vocab), num_hiddens=num_hiddens,
                    ffn_num_input=ffn_num_input,
                    ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                    num_layers=num_layers, dropout=dropout,
                    key_size=key_size, query_size=query_size, value_size=value_size,
                    norm_shape=norm_shape).to(device)
    decoder=TransformDecorder(vocab_size=len(tgt_vocab), num_hiddens=num_hiddens,
                    ffn_num_hiddens=ffn_num_hiddens,
                     ffn_num_input=ffn_num_input,
                      num_heads=num_heads
                    , num_layers=num_layers, dropout=dropout,
                    key_size=key_size, query_size=query_size, value_size=value_size,
                    norm_shape=norm_shape).to(device)
    transformer=Transformer(encoder,decoder).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    os.makedirs('models', exist_ok=True)
    best_loss = float('inf')
    existing_model_loss = None
    if os.path.exists('models/best_transformer.pth'):
        checkpoint = torch.load('models/best_transformer.pth', map_location=device, weights_only=True)
        existing_model_loss = checkpoint['loss']
        best_loss = existing_model_loss
        print(f"发现现有模型，其损失为: {existing_model_loss:.4f}")
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 可视化设置
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (per Epoch)")  # 修改标题为每轮损失
    line, = ax.plot([], [], label="Epoch Loss")  # 线条标签改为每轮损失
    ax.legend()

    # 存储每个epoch的损失和对应的epoch索引（用于绘图）
    epoch_losses = []  # 存储每个epoch的平均损失
    epoch_indices = []  # 存储对应的epoch索引（1,2,3...）

    # 训练循环
    for epoch in range(num_epochs):
        transformer.train()
        total_loss = 0.0
        batch_losses = []
        
        print(f"\nEpoch {epoch+1}/{num_epochs} 开始训练...")
        for batch in dataloader:
            src_X = batch['src_indices'].to(device)
            tgt_X = batch['tgt_indices'].to(device)
            tgt_Y = tgt_X[:, 1:].contiguous().reshape(-1)
            tgt_input = tgt_X[:, :-1].contiguous()
            
            optimizer.zero_grad()
            output = transformer(src_X, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            loss = loss_fn(output, tgt_Y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
        
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 结束，平均损失：{avg_epoch_loss:.4f}")
        
        # 每轮都记录损失和索引
        epoch_losses.append(avg_epoch_loss)
        epoch_indices.append(epoch + 1)  # 当前epoch编号（1-based）
        
        # 每轮都更新图像
        line.set_data(epoch_indices, epoch_losses)
        ax.relim()  # 重新计算坐标轴范围（关键：适应新数据）
        ax.autoscale_view()  # 自动调整视图范围
        plt.draw()  # 绘制更新
        plt.pause(0.01)  # 短暂暂停以刷新图像
        
        scheduler.step(avg_epoch_loss)
        
        if existing_model_loss is not None and avg_epoch_loss < existing_model_loss:
            print(f"当前模型损失({avg_epoch_loss:.4f})优于现有模型({existing_model_loss:.4f})")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'models/best_transformer.pth')
            print(f"  保存最佳模型（损失：{best_loss:.4f}）")

    # 训练完成后保存并显示最终图像
    plt.ioff()  # 关闭交互模式
    ax.set_title("Training Loss (Final)")
    plt.savefig("loss_curve.png")
    plt.show()
    print("\n训练完成！")
if __name__ == "__main__":
    train()