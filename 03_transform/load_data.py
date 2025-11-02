import os
import re
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

def read_translation_data(data_path):
    """读取双语平行语料（每行格式: 源语言句子\t目标语言句子\t版权信息）"""
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    # 提取前两列，忽略第三列版权信息，过滤空行
    pairs = [
        (parts[0], parts[1])  # 只取前两列
        for line in lines 
        if line.strip() and len(parts := line.split('\t')) >= 2  # 确保至少有两列
    ]
    src_sents, tgt_sents = zip(*pairs)  # 分离源语言和目标语言句子
    return list(src_sents), list(tgt_sents)

def tokenize(sentences, language):
    """
    对句子进行词元化
    :param sentences: 句子列表
    :param language: 语言类型（'en' 或 'zh' 等）
    :return: 词元列表的列表
    """
    tokenized = []
    for sent in sentences:
        # 英文：按空格分词，保留基本标点
        if language == 'en':
            # 更健壮的英文分词正则表达式
            tokens = re.findall(r"[\w']+|[.,!?;]", sent.lower())
        # 中文：按字符分词（可根据需要替换为jieba等分词工具）
        elif language == 'zh':
            # 改进中文分词，过滤可能的空白字符
            tokens = [char for char in sent if char.strip()]
        else:
            raise ValueError(f"不支持的语言: {language}")
        tokenized.append(tokens)
    return tokenized

class Vocab:
    """词表类：将词元映射到整数编码"""
    def __init__(self, tokens=None, min_freq=2, reserved_tokens=None):
        reserved_tokens = reserved_tokens or []
        if tokens is None:
            tokens = []
        # 统计词频
        counter = defaultdict(int)
        for line in tokens:
            for token in line:
                counter[token] += 1
        # 按词频筛选，保留高频词
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 构建词到索引的映射（预留特殊符号：<pad>, <bos>, <eos>, <unk>）
        self.itos = reserved_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        # 添加筛选后的词元
        for token, freq in self.token_freqs:
            if freq >= min_freq and token not in self.stoi:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, tokens):
        """将词元列表转换为索引列表（未知词元用<unk>）"""
        if not isinstance(tokens, (list, tuple)):
            return self.stoi.get(tokens, self.stoi['<unk>'])
        return [self.__getitem__(token) for token in tokens]


def process_tokens(tokens, vocab, max_len):
    """
    处理词元：添加特殊符号、截断或填充
    :param tokens: 词元列表
    :param vocab: 词表
    :param max_len: 最大序列长度
    :return: 处理后的索引列表和掩码
    """
    # 添加句首和句尾符号
    processed = ['<bos>'] + tokens + ['<eos>']
    
    # 截断过长序列
    if len(processed) > max_len:
        processed = processed[:max_len]
    
    # 填充短序列
    pad_len = max_len - len(processed)
    processed += ['<pad>'] * pad_len
    
    # 转换为索引
    indices = vocab[processed]
    
    # 创建掩码（1表示有效 token，0表示填充）
    mask = [1 if token != '<pad>' else 0 for token in processed]
    
    return indices, mask


class TranslationDataset(Dataset):
    """翻译数据集类"""
    def __init__(self, src_tokens, tgt_tokens, src_vocab, tgt_vocab, max_len=32):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        # 获取原始词元
        src_token = self.src_tokens[idx]
        tgt_token = self.tgt_tokens[idx]
        
        # 处理源语言和目标语言词元
        src_indices, src_mask = process_tokens(src_token, self.src_vocab, self.max_len)
        tgt_indices, tgt_mask = process_tokens(tgt_token, self.tgt_vocab, self.max_len)
        
        return {
            'src_indices': src_indices,
            'src_mask': src_mask,
            'tgt_indices': tgt_indices,
            'tgt_mask': tgt_mask,
            'src_raw': src_token,  # 原始词元，用于调试
            'tgt_raw': tgt_token
        }


def custom_collate_fn(batch):
    """自定义批量处理函数，处理不等长的原始词元列表"""
    # 分离各个字段
    src_indices = [item['src_indices'] for item in batch]
    src_mask = [item['src_mask'] for item in batch]
    tgt_indices = [item['tgt_indices'] for item in batch]
    tgt_mask = [item['tgt_mask'] for item in batch]
    src_raw = [item['src_raw'] for item in batch]  # 保留原始词元列表
    tgt_raw = [item['tgt_raw'] for item in batch]
    
    # 将数值型数据转换为张量
    return {
        'src_indices': torch.tensor(src_indices, dtype=torch.long),
        'src_mask': torch.tensor(src_mask, dtype=torch.float),
        'tgt_indices': torch.tensor(tgt_indices, dtype=torch.long),
        'tgt_mask': torch.tensor(tgt_mask, dtype=torch.float),
        'src_raw': src_raw,  # 直接返回列表（不转换为张量）
        'tgt_raw': tgt_raw
    }


# 示例用法
if __name__ == "__main__":
    # 假设数据文件格式：每行 "英文句子\t中文句子"
    data_path = "/home/Elaina/pytorch/Data/cmn.txt"
    src_lang, tgt_lang = 'en', 'zh'
    batch_size = 4  # 小批量大小
    max_seq_len = 16  # 最大序列长度
    
    # 读取数据
    src_sents, tgt_sents = read_translation_data(data_path)
    print(f"读取到 {len(src_sents)} 条平行语料")
    
    # 词元化
    src_tokens = tokenize(src_sents, src_lang)
    tgt_tokens = tokenize(tgt_sents, tgt_lang)
    print(f"示例词元化（英文）：{src_tokens[0]}")  # 应该输出 ['hi', '.']
    print(f"示例词元化（中文）：{tgt_tokens[0]}")  # 应该输出 ['嗨', '。']
    
    # 构建词表
    reserved = ['<pad>', '<bos>', '<eos>', '<unk>']  # 特殊符号：填充、句首、句尾、未知
    src_vocab = Vocab(src_tokens, min_freq=1, reserved_tokens=reserved)  # 测试时将min_freq设为1
    tgt_vocab = Vocab(tgt_tokens, min_freq=1, reserved_tokens=reserved)
    print(f"源语言词表大小：{len(src_vocab)}，目标语言词表大小：{len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(
        src_tokens, tgt_tokens, 
        src_vocab, tgt_vocab, 
        max_len=max_seq_len
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,  # 打乱数据顺序
        drop_last=True,  # 丢弃最后一个不完整的批次
        collate_fn=custom_collate_fn  # 使用自定义的批量处理函数
    )
    
    # 输出一个小批量数据示例
    print("\n--- 小批量数据示例 ---")
    for batch in dataloader:
        # 打印批次信息
        print(f"\n批次大小: {batch['src_indices'].shape[0]}")
        print(f"源语言编码形状: {batch['src_indices'].shape}")
        print(f"目标语言编码形状: {batch['tgt_indices'].shape}")
        
        # 打印每条数据的详细信息
        for i in range(batch_size):
            print(f"\n样本 {i+1}:")
            print(f"源语言原始词元: {batch['src_raw'][i]}")
            print(f"源语言编码: {batch['src_indices'][i].tolist()}")
            print(f"源语言掩码: {batch['src_mask'][i].tolist()}")
            print(f"目标语言原始词元: {batch['tgt_raw'][i]}")
            print(f"目标语言编码: {batch['tgt_indices'][i].tolist()}")
            print(f"目标语言掩码: {batch['tgt_mask'][i].tolist()}")
        
        # 只展示一个批次
        break