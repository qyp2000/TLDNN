from yacs.config import CfgNode as CN
import argparse
import importlib

__all__ = ['get_cfgs']

_C = CN()

_C.modes = CN()
_C.modes.method = 'DAELSTM'
_C.modes.path = ''
_C.modes.loss = 'loss_CE_and_MSE'
_C.modes.train = False
_C.modes.ddp = False

_C.data_settings = CN()
_C.data_settings.dataset = 'RML2016'
_C.data_settings.Xmode = CN()
_C.data_settings.Xmode.type = 'IQ'
_C.data_settings.Xmode.options = CN()
_C.data_settings.Xmode.options.IQ_norm = False
_C.data_settings.Xmode.options.zero_mask = False
_C.data_settings.Xmode.options.random_mix = False
_C.data_settings.mod_type = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'PAM4', 'WBFM', 'AM-SSB', 'AM-DSB']

_C.opt_params = CN()
_C.opt_params.batch_size = 400
_C.opt_params.epochs = 150
_C.opt_params.lr = 1e-2
_C.opt_params.workers = 0
_C.opt_params.seed = 1
_C.opt_params.gpu = 0
_C.opt_params.cpu = False
_C.opt_params.early_stop = False


def get_cfg_defaults():
    return _C.clone()


def get_cfgs():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='configs/transformerlstm_16.yaml',
                        help='type of config file. e.g. transformerlstm_16 (configs/transformerlstm_16.yaml)')
    opt, unparsed = parser.parse_known_args()
    cfgs.merge_from_file(opt.config)
    return cfgs
