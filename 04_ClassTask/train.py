from transformer import *
from load_data import *
from visualAndSave import *
import torch, os
import torch.optim as optim
import torch.nn as nn

def train():
    # 超参数设置
    lr, num_epochs,batch_size,seq_min_len ,device = 0.001, 500, 80,7,torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #编码器序列长度，特征长度，解码器特征长度，隐藏单元数
    seq_len=20
    # 数据加载
    file_path = "/pytorch/Data/data.xlsx"
    # 加载原始数据
    encoder_inputs, decoder_inputs, decoder_outputs = [],[],[]
    #增加sheet1-4的数据
    for i in range(4):
        sheet_name='Sheet'+str(i+1)
        once_encoder_inputs,once_decorder_inputs,once_decoder_outputs=LoadData(file_path,7,sheet_name)
        encoder_inputs+=once_encoder_inputs
        decoder_inputs+=once_decorder_inputs
        decoder_outputs+=once_decoder_outputs
    # #初始化模型
    # encoder=Encoder(feature_size=encoder_feature_size,key_size=num_hiddens,query_size=num_hiddens,value_size=num_hiddens,num_hiddens=num_hiddens,norm_shape=[num_hiddens],ffn_num_input=num_hiddens,ffn_num_hiddens=ffn_num_hiddens,num_heads=num_heads,num_layers=num_layer).to(device)
    # decoder=TransformDecoder(feature_size=decoder_feature_size,key_size=num_hiddens,query_size=num_hiddens,value_size=num_hiddens,num_hiddens=num_hiddens,norm_shape=[num_hiddens],ffn_num_input=num_hiddens,ffn_num_hiddens=ffn_num_hiddens,num_heads=num_heads,num_layers=num_layer).to(device)
    # transformer = Transformer(encoder, decoder).to(device)
    transformer=LightweightAttention(hidden_dim=8).to(device)
    # 平方损失
    loss_fn = AmplifiedResidualLoss().to(device)
    # 优化器与学习率调度器
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 初始化模型管理器
    model_manager = SaveAndVisual(model_dir='/pytorch/models/task.pth',loss_img_path='/pytorch/models/'+'loss_curve.png')
    # best_loss = model_manager.load_model(transformer, optimizer, device)
    model_manager.loadModel(transformer, optimizer, device)
    # 训练循环
    for epoch in range(num_epochs):
        transformer.train()
        enc_batch, dec_in_batch, dec_out_batch = create_batch(
                encoder_inputs,
                decoder_inputs,
                decoder_outputs,
                encoder_seq_len=seq_len,
                seq_len=seq_min_len,
                batch_size=batch_size
            )
        #将解码器的输入去掉尾，输出去掉头
        dec_in_batch=dec_in_batch[:,:-1,:]
        dec_out_batch=dec_out_batch[:,1:,:]
        #移动到设备
        enc_batch, dec_in_batch, dec_out_batch = enc_batch.to(device), dec_in_batch.to(device), dec_out_batch.to(device)
        #前向传播
        outputs,weights = transformer(dec_in_batch, enc_batch)
        loss = loss_fn(outputs, dec_out_batch)
        model_manager.updateVisualization(epoch, loss.item())
        #反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 学习率调度
        # scheduler.step(loss)
    model_manager.finalizeVisualization()


        

    # 训练结束处理
    # model_manager.finalize_visualization()
    print("\n训练完成！")


if __name__ == "__main__":
    train()