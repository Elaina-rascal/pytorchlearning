from transformer import *
from load_data import *
import torch,os
import torch.optim as optim
import torch.nn as nn

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
    # 训练循环
    for epoch in range(num_epochs):
        transformer.train()
        total_loss = 0.0
        batch_count = 0  # 用于计数批次
        
        print(f"\nEpoch {epoch+1}/{num_epochs} 开始训练...")
        for batch in dataloader:
            # 数据预处理
            src_X = batch['src_indices'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_X = batch['tgt_indices'].to(device)
            src_valid_lens = src_mask.sum(dim=1).to(device)  # 适用于 True=有效 的掩码
            # 目标序列偏移（教师强制）
            tgt_Y = tgt_X[:, 1:].contiguous()  # 目标：去掉第一个token（如<BOS>）
            tgt_input = tgt_X[:, :-1].contiguous()  # 解码器输入：去掉最后一个token
            
            # 前向传播
            optimizer.zero_grad()  # 清零梯度
            output:torch.Tensor= transformer(src_X, tgt_input,src_valid_lens)  # 模型输出
            
            # 计算损失
            output = output.reshape(-1, output.shape[-1])  # 形状调整为 (batch*seq_len, vocab_size)
            tgt_Y = tgt_Y.reshape(-1)  # 形状调整为 (batch*seq_len,)
            loss = loss_fn(output, tgt_Y)
            
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            batch_count += 1
            
            # 每10个批次打印一次当前损失（可选）
            if batch_count % 10 == 0:
                print(f"  批次 {batch_count}/{len(dataloader)}，当前损失：{loss.item():.4f}")
        
        # 计算本轮平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 结束，平均损失：{avg_loss:.4f}")
        
        # 学习率调整
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'models/best_transformer.pth')
            print(f"  保存最佳模型（损失：{best_loss:.4f}）")
    
    print("\n训练完成！")

if __name__ == "__main__":
    train()