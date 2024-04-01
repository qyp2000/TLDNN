from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import math
from torch import Tensor
from amr.dataloaders.dataloader2 import *
from amr.utils import *
from thop import profile                            
from thop import clever_format
from ptflops import get_model_complexity_info

__all__ = ['model_complexity']

def model_complexity(cfgs, net):
    batchsize = 1
    if cfgs.data_settings.dataset == 'RML2016':
      data_input = Variable(torch.randn([batchsize, 2, 128]))
    elif cfgs.data_settings.dataset == 'RML2018':
      data_input = Variable(torch.randn([batchsize, 2, 1024]))
    net.eval()
    
    if cfgs.modes.method == "TransformerMultiLoss":
      print(flop_count_table(FlopCountAnalysis(net, (data_input,data_input))))
    else:
      print(flop_count_table(FlopCountAnalysis(net, data_input)))
      #flops, params = profile(net, inputs=(data_input, ))                    
      #flops,params = clever_format([flops, params],"%.3f")
      #print(params,flops) 
      #acs, params_ptflops = get_model_complexity_info(net, (2, data_input.shape[2]), print_per_layer_stat=True)
     


if __name__ == '__main__':
    cfgs = get_cfgs()
    net = init_model(cfgs)
    model_complexity(cfgs, net)

