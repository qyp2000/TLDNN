from torch.autograd import Variable
import torch
from torchsummaryX import summary
import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from thop import clever_format
from einops import rearrange, repeat
__all__ = ["TransformerLSTM"]


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class FeedForward_GLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.ReLU()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.gelu(x1)*x2
        f = self.fc3(x)
        return f

class FeedForward_2headsGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1*self.gelu(x2)
        f2 = x2*self.gelu(x1)
        f = self.fc3(f1+f2)
        return f

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0., talk_heads=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.talking_heads1 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()
        self.talking_heads2 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = self.talking_heads1(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn = self.talking_heads2(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, talk_heads, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, talk_heads=talk_heads),
                #nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False),
                nn.LayerNorm(dim),
                FeedForward_GLU(dim, mlp_dim, dropout=dropout),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x):
        for attn, norm1, ff, norm2 in self.layers:
            x = norm1(attn(x) + x)
            #x = norm1(attn(x,x,x,need_weights=False, average_attn_weights=False)[0] + x)
            x = norm2(ff(x) + x)
        return x

class GroupBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(GroupBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=5, padding=2, groups=in_channel),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, in_channel, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Identity(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x3 = self.relu(x1 + x2)
        return x3

    def __call__(self, x):
        return self.forward(x)


class TransformerLSTM(nn.Module):
    def __init__(self, classes: int, d_model: int = 64, nhead: int = 8, d_hid: int = 128, lstm_hidden_size: int = 64,
                 nlayers: int = 2 , dropout: float = 0.2, max_len: int = 5000, mask=False, poolsize = (2,1)):
        super().__init__()
        self.model_type = 'TransformerLSTM'
        self.mask = mask
        self.d_model = d_model

        self.Convlayer = nn.Sequential(
                    nn.Conv1d(2, d_model, kernel_size = 4, stride = 2, padding = 1),
                    nn.ReLU(),
                    nn.Conv1d(d_model, d_model, kernel_size = 4, stride = 2, padding = 1),
                    )
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, d_model))
        self.se = SEAttention(channel=d_model, reduction=4)
        self.dropout_layer = nn.Dropout(dropout/2)

        self.transformer_encoder =  nn.Sequential(
                    Transformer(dim=d_model, depth=nlayers, heads=nhead, dim_head=d_model//nhead, mlp_dim=d_hid, talk_heads=True, dropout=dropout),
                    )

        self.lstm_layer = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_size, num_layers=4, dropout=dropout, batch_first=True)        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size),
            nn.ReLU(),  # 激活函数
            nn.BatchNorm1d(lstm_hidden_size),  # 批量归一化,
            nn.Dropout(dropout),
            
            nn.Linear(in_features=lstm_hidden_size, out_features=int(lstm_hidden_size/2)),
            nn.ReLU(),  # 激活函数
            nn.BatchNorm1d(int(lstm_hidden_size/2)),  # 批量归一化
            nn.Dropout(dropout),

            nn.Linear(in_features=int(lstm_hidden_size/2), out_features=classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.Convlayer(x).transpose(1, 2)
        B, N, _ = x.shape

        x += self.pos_embedding[:, :N]
        x = x.transpose(1,2)
        x = self.se(x)
        x = self.dropout_layer(x)

        #Batchsize Embbdingdim Seqlen
        x = self.transformer_encoder(x.transpose(1,2))
        dec_out, (hidden_state, cell_state) = self.lstm_layer(x)
        fc = self.decoder(hidden_state[-1])

        return fc

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if __name__ == '__main__':
    net = TransformerLSTM(11)
    batchsize = 1
    data_input = Variable(torch.randn([batchsize, 2, 128]))
    #summary(net, data_input)
    #net(data_input)
    net.eval()
    print(flop_count_table(FlopCountAnalysis(net, data_input)))
    flops, params = profile(net, inputs=(data_input, ))
    flops,params = clever_format([flops, params],"%.3f")
    print(params,flops) 
