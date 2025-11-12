from transformer import *
from load_data import *
from visualAndSave import *
import torch, os
import torch.optim as optim
import torch.nn as nn

def train():
    # 超参数设置
    num_hiddens, num_layers, dropout, batch_size = 64, 4, 0.1, 64
    lr, num_epochs, device = 0.005, 200, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ffn_num_input, ffn_num_hiddens, num_heads = 64, 128, 4
    key_size, query_size, value_size = 64, 64, 64
    norm_shape = [64]

    # 数据加载
    data_path = "/pytorch/Data/cmn.txt"
    src_lang, tgt_lang = 'en', 'zh'
    src_sents, tgt_sents = read_translation_data(data_path)
    print(f"读取到 {len(src_sents)} 条平行语料")
    
    # 词元化与词表构建
    src_tokens = tokenize(src_sents, src_lang)
    tgt_tokens = tokenize(tgt_sents, tgt_lang)
    
    reserved = ['<pad>', '<bos>', '<eos>', '<unk>']
    src_vocab = Vocab(src_tokens, min_freq=1, reserved_tokens=reserved)
    tgt_vocab = Vocab(tgt_tokens, min_freq=1, reserved_tokens=reserved)
    print(f"源语言词表大小：{len(src_vocab)}，目标语言词表大小：{len(tgt_vocab)}")
    
    # 数据集与数据加载器
    dataset = TranslationDataset(src_tokens, tgt_tokens, src_vocab, tgt_vocab)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    # 模型初始化
    encoder = Encoder(
        vocab_size=len(src_vocab), num_hiddens=num_hiddens,
        ffn_num_input=ffn_num_input, ffn_num_hiddens=ffn_num_hiddens,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        key_size=key_size, query_size=query_size, value_size=value_size,
        norm_shape=norm_shape
    ).to(device)
    
    decoder = TransformDecorder(  # 注意：原代码中可能是TransformDecoder，这里保持原样
        vocab_size=len(tgt_vocab), num_hiddens=num_hiddens,
        ffn_num_hiddens=ffn_num_hiddens, ffn_num_input=ffn_num_input,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        key_size=key_size, query_size=query_size, value_size=value_size,
        norm_shape=norm_shape
    ).to(device)
    
    transformer = Transformer(encoder, decoder).to(device)

    # 优化器与损失函数
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 初始化模型管理器
    model_manager = SaveAndVisual(model_dir='/pytorch/models',loss_img_path='/pytorch/models'+'loss_curve.png')
    best_loss = model_manager.load_model(transformer, optimizer, device)
    existing_model_loss = best_loss

    # 训练循环
    for epoch in range(num_epochs):
        transformer.train()
        total_loss = 0.0
        
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
        
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 结束，平均损失：{avg_epoch_loss:.4f}")
        
        # 更新可视化
        model_manager.update_visualization(epoch + 1, avg_epoch_loss)
        
        # 学习率调度
        scheduler.step(avg_epoch_loss)
        
        # 模型保存判断
        if existing_model_loss is not None and avg_epoch_loss < existing_model_loss:
            print(f"当前模型损失({avg_epoch_loss:.4f})优于现有模型({existing_model_loss:.4f})")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model_manager.save_model(transformer, optimizer, epoch, best_loss)

    # 训练结束处理
    model_manager.finalize_visualization()
    print("\n训练完成！")


if __name__ == "__main__":
    train()