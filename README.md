# README
This is the PyTorch implementation of paper “Enhancing Automatic Modulation Recognition through Robust Global Feature Extraction”.

# Requirements
```
pytorch
yacs
h5py
pickle
matplotlib
thop
ptflops
fvcore
```

# Architecture
``` 
home
├── amr/
│   ├── dataloaders/
│   ├── models/
│   │   ├── losses/
│   │   ├── networks/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── draw.py
│   │   ├── init.py
│   │   ├── logger.py
│   │   ├── solver.py
│   │   ├── static.py
├── configs/
│   ├── *.yaml
├── main.py
├── complexity.py
├── datasets/
├── results/
```

# Train and Test Your Own Model
1. preparing dataset: download the datasets RadioML2016.10a and RadioML2018.01a on [DeepSig's official website](https://www.deepsig.ai/datasets), and form the file path like `dataset/RML2016.10a_dict.pkl` and `dataset/RML2018.01.hdf5`.

2. training and testing the model: run `python main.py --config xxx`. e.g.`python main.py --config configs/transformerlstm_18.yaml`
We have devised a training strategy using DistributedDataParallel (DDP). To train using the DDP approach, set the parameter `ddp` to `True` in the config and run
`CUDA_VISIBLE_DEVICES=xxx python -m torch.distributed.launch --nproc_per_node=xxx main.py --config configs/transformerlstm_18.yaml`. e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/transformerlstm_18.yaml`

3. checking the results: check the well-trained models and some visualized results in `results/`. 


# Result Reproduction
1. preparing dataset: download the datasets RadioML2016.10a and RadioML2018.01a on [DeepSig's official website](https://www.deepsig.ai/datasets), and form the file path like 'dataset/RML2016.10a_dict.pkl' and 'dataset/RML2018.01.hdf5'.

2. preparing models: We save the well-trained weights of TLDNN in `results/TransformerLSTM/best/RML2016/checkpoints/best_acc.pth` for RadioML2016.10a and 
`results/TransformerLSTM/best/RML2018/checkpoints/best_acc.pth` for RadioML2018.01a. Additionally, some of the visualized results have also been saved.
The prepared models can be used directly.

3. modifying settings: set the parameter `train` From `True` to `False` in `configs/transformerlstm_16.yaml` and `configs/transformerlstm_18.yaml`. Then run `python main.py --config configs/transformerlstm_16.yaml` and `python main.py --config configs/transformerlstm_18.yaml` to get the expermential results.

    e.g.
    ```yaml for RadioML2016.10a
    modes:  
        method: 'TransformerLSTM'
        path: 'best' # save address
        loss: 'loss_CE'
        train: False # set the parameter: from True to False
        ddp: False # choose whether to use ddp trainint
    data_settings:  
        dataset: 'RML2016'
        Xmode:
            type: 'AP' # input type for the model
            options:
                IQ_norm: False # choose whether to normalized the input signal
                zero_mask: False
                random_mix: False #choose whether to use the random mixing strategy
        mod_type: ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    opt_params: 
        batch_size: 128
        epochs: 150
        lr: 1e-3
        workers: 8
        seed: 1
        gpu: 0
        cpu: False
        early_stop: False
    ```

    ```yaml for RadioML2018.01a
    modes:  
        method: 'TransformerLSTM'
        path: 'best'
        loss: 'loss_CE'
        train: False # set the parameter: from True to False
        ddp: False
    data_settings:  
        dataset: 'RML2018'
        Xmode:
            type: 'AP'
            options:
                IQ_norm: False 
                zero_mask: False
                random_mix: False
        mod_type: [ '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC',
        '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
        'AM-DSB-WC', 'OOK', '16QAM' ]
    opt_params:  
        batch_size: 400
        epochs: 150
        lr: 1e-3
        workers: 8
        seed: 1
        gpu: 0
        cpu: False
        early_stop: False
    ```




