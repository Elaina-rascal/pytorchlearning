from torch import nn
import torch
from transformer import *
from load_data import *
def infer(transformer:Transformer, src_sentence:str, src_vocab:Vocab, tgt_vocab:Vocab, device:torch.device, max_len:int=32)->str:
    """使用训练好的Transformer模型进行翻译推断"""
    # 词元化
    src_tokens = tokenize([src_sentence], 'en')[0]
    # 处理源语言词元
    src_indices, src_mask = process_tokens(src_tokens, src_vocab, max_len)
    src_X = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
    src_valid_lens = torch.tensor([sum(src_mask)], dtype=torch.long).to(device)  # 有效长度

    # 编码器前向传播
    enc_outputs = transformer.encoder(src_X, src_valid_lens)
    
    # 初始化解码器输入，添加句首标记
    tgt_indices = [tgt_vocab['<bos>']]
    for _ in range(max_len):
        tgt_X = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
        dec_state = transformer.decoder.init_state(enc_outputs, src_valid_lens)
        
        # 解码器前向传播
        dec_outputs = transformer.decoder(tgt_X, dec_state)
        
        # 取最后一个时间步的输出进行预测
        pred = dec_outputs[:, -1, :].argmax(dim=-1).item()
        tgt_indices.append(pred)
        
        # 遇到句尾标记则停止生成
        if pred == tgt_vocab['<eos>']:
            break
    
    # 转换为词元并返回翻译结果
    translated_tokens= tgt_vocab.to_tokens(tgt_indices)
    return ' '.join(translated_tokens)
def main():
    num_hiddens, num_layers, dropout,  = 32, 2, 0.1, 
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载词表
    data_path = "/home/Elaina/pytorch/Data/cmn.txt"
    src_lang, tgt_lang = 'en', 'zh'
    src_sents, tgt_sents = read_translation_data(data_path)
    print(f"读取到 {len(src_sents)} 条平行语料")
    reserved = ['<pad>', '<bos>', '<eos>', '<unk>']  # 特殊符号：填充、句首、句尾、未知
    # 词元化
    src_tokens = tokenize(src_sents, src_lang)
    tgt_tokens = tokenize(tgt_sents, tgt_lang)
    src_vocab = Vocab(src_tokens, min_freq=1, reserved_tokens=reserved)  # 测试时将min_freq设为1
    Target_vocab = Vocab(tgt_tokens, min_freq=1, reserved_tokens=reserved)
    input=["you","are","fucking","stupid","."]  #测试句子

    #加载transformer与权重
    encoder=Encoder(vocab_size=len(src_vocab), num_hiddens=num_hiddens,
                    ffn_num_input=ffn_num_input,
                    ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                    num_layers=num_layers, dropout=dropout,
                    key_size=key_size, query_size=query_size, value_size=value_size,
                    norm_shape=norm_shape).to(device)
    decoder=TransformDecorder(vocab_size=len(Target_vocab), num_hiddens=num_hiddens,
                    ffn_num_hiddens=ffn_num_hiddens,
                     ffn_num_input=ffn_num_input,
                      num_heads=num_heads
                    , num_layers=num_layers, dropout=dropout,
                    key_size=key_size, query_size=query_size, value_size=value_size,
                    norm_shape=norm_shape).to(device)
    transformer=Transformer(encoder,decoder).to(device)
    checkpoint = torch.load('models/best_transformer.pth', map_location=device, weights_only=True)  # 加上weights_only=True避免警告
    transformer.load_state_dict(checkpoint['model_state_dict'])  # 只加载模型参数部分
    
    # 进行翻译推断
    test_sentence = "You JJ B ."
    translation = infer(transformer, test_sentence, src_vocab, Target_vocab, device)
    print(f"源句子: {test_sentence}")
    print(f"翻译结果: {translation}")
if __name__ == "__main__":
    main()