import torch
import torch.nn as nn
import torch.nn.functional as F


#每次只读一个句子中的一个字母，感觉没有时间步呀,估计时间步用作循环处理了
#单独的编码器
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):  #input_size 表示输入词的词汇表大小 hidden_size表示输出隐藏层
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x,h0):
        x = self.embedding(x)
        #print(x)
        x = x.view(1,1,-1)
        #print(x.shape)
        output, hi = self.gru(x,h0)
        return output, hi
    def init_hidden(self,batch_size = 1,num_layers=1):
        return torch.zeros(num_layers,batch_size,self.hidden_size)
#单独的解码器
class DecoderGRU(nn.Module):
    def __init__(self, output_size, hidden_size):  #output_size 表示输出词的大小这样才好匹配词汇表 hidden_size表示输出隐藏层
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = output_size
        self.embedding = nn.Embedding(output_size, 30)
        self.gru = nn.GRU(30, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x,h0):
        x = self.embedding(x)
        print(x.shape)
        x = x.view(1,1,-1)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        output, hi = self.gru(x,h0)
        print(output.shape)
        output = self.out(output)
        print(output.shape)
        output = output.squeeze(0)
        print(output.shape)
        output = self.softmax(output)
        print(output.shape)
        return output, hi
    def init_hidden(self,batch_size = 1,num_layers=1):
        return torch.zeros(num_layers,batch_size,self.hidden_size)


class attention_decoder(nn.Module):
    def __init__(self, hidden_size, output_size: int,max_length):
        super(attention_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.attention = nn.Linear(hidden_size * 2, max_length)
        self.attention2 = nn.Linear(hidden_size * 2, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x,encoder_outputs):
        x = self.embedding(x)
        # print(x)
        # print(x.shape)
        x = self.dropout(x)
        # print(x)
        # print(x.shape)
        """
        所谓的注意力机制的应用，就是把所有的out_puts加权平均后给我，我好挑
        """
        #这个地方不好理解 x 和 h0组成权重矩阵但我想这个矩阵是可学习的就加了个线性层，也就是所谓的注意力层
        x_atten = self.attention(torch.cat((x,encoder_outputs[-1].unsqueeze(0)),dim=-1))

        # print(x_atten)
        # print(x_atten.shape)
        x_atten = F.softmax(x_atten,dim=-1)
        # print(x_atten)
        # print(x_atten.shape)
        #我希望编码器不要把所有输出原封不动地给我即每个h相加给我，我想要权重分配后相加给我
        encoder_outputs_atten = x_atten @ encoder_outputs
        # print(encoder_outputs_atten)
        # print(encoder_outputs_atten.shape)
        """
        加权后的上文也有了，我怎么用这个呢   还是注意力机制
        """
        input = torch.cat([x,encoder_outputs_atten],dim=-1)
        # print(input)
        # print(input.shape)
        input = self.attention2(input)
        # print(input)
        # print(input.shape)
        input = F.relu(input)
        output,hidden = self.GRU(input.unsqueeze(0),encoder_outputs[-1].unsqueeze(0).unsqueeze(0))
        # print(hidden)
        # print(hidden.shape)
        output = F.log_softmax(self.out(output).squeeze(0),dim = -1)  #预测概率
        # print(output)
        # print(output.shape)
        return output,hidden,encoder_outputs_atten





if __name__ == '__main__':
    #encoder测试用例
    # mygru = EncoderGRU(20,32)
    # x = torch.tensor([2])
    # h0 = mygru.init_hidden()
    # output, hn = mygru(x,h0)
    # print(output)
    # print(hn)
    # #decoder测试用例
    # mygru = DecoderGRU(32, 25)
    # x = torch.tensor([10])
    # h0 = mygru.init_hidden()
    # output, hn = mygru(x, h0)
    # print(output)
    # print(hn.shape)
    x = torch.tensor([2])
    encoder_outputs = torch.ones(10,25)
    mymodel = attention_decoder(hidden_size=25,output_size=256,max_length=10)
    output,hidden,encoder_outputs_atten = mymodel(x, encoder_outputs)
    print(output.shape)
    print(hidden.shape)
    print(encoder_outputs_atten.shape)