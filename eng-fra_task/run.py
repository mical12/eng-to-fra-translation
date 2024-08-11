import random

from matplotlib import pyplot as plt

import data_construction
import model
import torch
import torch.nn as nn
import time
import train_data

def run(iters, my_encoder, my_decoder, criterion, optimizer1, optimizer2,lang1, lang2, pairs):
    plot_losses = 0
    print_losses = 0
    history_loss = []
    start = time.time()
    for iter in range(1,iters+1):
        input, output = data_construction.tensor_from_sequence(lang1, lang2, random.choice(pairs))
        loss = train_data.train(input, output, my_encoder, my_decoder, criterion, optimizer1, optimizer2)
        plot_losses += loss
        print_losses += loss

        #到一定步数时打印
        if iter %100 == 0:
            #打印固定批次的日志以及为后续画图做准备
            end = time.time()
            plot_losses_aveg = plot_losses / 100
            history_loss.append(plot_losses_aveg)
            print_losses = 0
            print(f"当前已迭代：{iter} 损失为{plot_losses_aveg} 进度为{iter/iters} 训练耗时为{end-start}s")
            plot_losses = 0
    plt.figure()
    plt.plot(history_loss)
    plt.savefig('loss.png')


if __name__ == '__main__':
    #取总的数据
    lang1, lang2, pairs = data_construction.prepare_data('eng', 'fra')
    #定义编码器解码器
    my_encoder = model.EncoderGRU(lang1.num_words, hidden_size=32)
    my_decoder = model.attention_decoder(32, lang2.num_words, max_length=10)
    #定义优化器
    criterion = nn.NLLLoss()
    optimizer1 = torch.optim.SGD(my_encoder.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(my_decoder.parameters(), lr=0.01)
    run(20000,my_encoder,my_decoder,criterion,optimizer1,optimizer2,lang1, lang2, pairs)
    # 保存模型
    path = './model.pth'
    torch.save(my_encoder.state_dict(), path)
    path = './model2.pth'
    torch.save(my_decoder.state_dict(), path)


