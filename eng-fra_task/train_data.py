import random
import time

import data_construction
import model
import torch
import torch.nn as nn

def train(input_tensor,output_tensor,my_encoder,my_decoder,criterion,optimizer1,optimizer2,hidden_size = 32,max_length = 10 ):#n行一列的数据
    teacher_forceing_rate = 0.5
    my_decoder.zero_grad()
    my_encoder.zero_grad()
    """
    构建编码器"""
    h0 = my_encoder.init_hidden()
    outputs_all = torch.zeros(max_length,hidden_size)
    i=0
    for input_word in input_tensor:
        output,hn = my_encoder(input_word,h0)  #1*1*32
        # print(output)
        #print(output.shape)
        #print(hn.shape)
        outputs_all[i] = hn.squeeze(0).squeeze(0)  #注意这里怎么取各隐藏层的张量
        i=i+1
    """
    构建解码器"""
    loss = 0
    decode_innit_word = torch.tensor([0])
    #使用teacher force技术
    use_teacher_forcing = True if random.random() <= teacher_forceing_rate else False
    if use_teacher_forcing:
        for target_word in output_tensor:
            decoder_output,hn,encoder_outputs_atten = my_decoder(decode_innit_word,outputs_all)
            loss += criterion(decoder_output, target_word)
            decode_innit_word = target_word
    else:
        for target_word in output_tensor:
            decoder_output,hn,encoder_outputs_atten = my_decoder(decode_innit_word,outputs_all)
            loss += criterion(decoder_output, target_word)
            #如果有一个预测为结束那么就终止预测了
            topv,topi = decoder_output.topk(1)
            if topi.squeeze().item() == 1:
                break
            decode_innit_word = topi[0].detach()
    #反向传播
    loss.backward()
    #更新梯度
    optimizer1.step()
    optimizer2.step()
    return loss.item()/output_tensor.size(0)

if __name__ == '__main__':
    lang1, lang2, pairs = data_construction.prepare_data('eng', 'fra')
    input, output = data_construction.tensor_from_sequence(lang1, lang2, pairs[0])
    my_encoder = model.EncoderGRU(lang1.num_words, hidden_size = 32)
    my_decoder = model.attention_decoder( 32, lang2.num_words, max_length = 10)
    criterion = nn.NLLLoss()
    optimizer1 = torch.optim.Adam(my_encoder.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(my_decoder.parameters(), lr=0.001)
    start = time.time()
    loss = train(input, output,my_encoder,my_decoder,criterion,optimizer1,optimizer2)
    ends = time.time()
    print(loss)

