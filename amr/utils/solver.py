from amr.utils import logger
import torch
import time
from amr.utils.static import *
import os
import numpy as np
__all__ = ["Trainer", "Tester"]

class Trainer:
    def __init__(self, model, device, optimizer, scheduler, criterion, save_path, valid_freq=1, early_stop=True, mask_flag=True, random_mix_flag=True, Random_Matrix = None, loss_name = None, snrs=None, model_name = None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.all_epoch = None
        self.cur_epoch = 1
        self.train_loss = None
        self.train_acc = None
        self.valid_loss = None
        self.valid_acc = None
        self.scheduler_val_loss = 10
        self.train_loss_all = []
        self.train_acc_all = []
        self.valid_loss_all = []
        self.valid_acc_all = []
        self.valid_freq = valid_freq
        self.save_path = save_path
        self.best_acc = None
        # early stopping
        self.early_stop = early_stop
        self.patience = 50
        self.delta = 0
        self.counter = 0
        self.stop_flag = False
        self.mask_flag = mask_flag
        self.loss_name = loss_name
        self.model_name = model_name
        self.random_mix_flag = random_mix_flag
        self.snrs = snrs
        self.p = 0.0625/4
        if random_mix_flag == False:
            self.Random_Matrix = Random_Matrix
        else:
            self.Random_Matrix = torch.tensor(Random_Matrix).to(self.device).float()
        print("mask_flag = ", mask_flag)
        print("random_mix_flag = ", random_mix_flag)
        print("p = ", self.p)
        print("early_stop = ", early_stop)

    def loop(self, epochs, train_loader, valid_loader):
        self.all_epoch = epochs
        for ep in range(self.cur_epoch, epochs+1):
            self.cur_epoch = ep
            self.train_loss, self.train_acc = self.train(train_loader)
            self.train_loss_all.append(self.train_loss)
            self.train_acc_all.append(self.train_acc)

            # if ep % self.valid_freq == 0:
            self.valid_loss, self.valid_acc = self.val(valid_loader)
            self.valid_loss_all.append(self.valid_loss)
            self.valid_acc_all.append(self.valid_acc)

            self._loop_postprocessing(self.valid_acc)

            if self.early_stop and self.stop_flag:
                logger.info(f'early stopping at Epoch: [{self.cur_epoch}]')
                break
        return self.train_loss_all, self.train_acc_all, self.valid_loss_all, self.valid_acc_all

    def get_iq_framed(self, X, L=32, R=16):
        # [2, 1024]
        F=int((X.shape[2]-L)/R+1)
        Y = torch.zeros([F, X.shape[0], 2*L]).to(self.device)
        i = 0
        for idx in range(0, X.shape[-1]-L+1, R):
            Y[i, :, :] = X[:, :, idx:idx+L].reshape([1,X.shape[0], 2*L])
            i = i+1
            #Y.append(X[:, :, idx:idx+L].reshape([1,X.shape[0], 2*L]))  # (2, L=32)
        #Y = np.vstack(Y)  # (F, 2L) = (63, 64)  F=(1024-L)/R+1
        Y = Y.permute(1,0,2)
        return Y

    def zero_mask(self, X_train, p=0.1):
        num = int(X_train.shape[2] * p)
        res = X_train.clone()
        index = np.array([[i for i in range(X_train.shape[2])] for _ in range(X_train.shape[0])])
        for i in range(index.shape[0]):
            np.random.shuffle(index[i, :])
        for i in range(res.shape[0]):
            res[i, :, index[i, :num]] = 0
        
        return res

    def continuous_random_mixing(self, X_train, Y_train, Z_train, p=0.0625, low_snr=True):
        num = int(X_train.shape[2] * p)
        res = X_train.clone()
        col_index = np.array([[i for i in range(num)] for _ in range(X_train.shape[0])])
        row_index = np.array([[i for i in range(X_train.shape[0])] for _ in range(num)]).transpose()
        choice_res = np.random.choice(range(X_train.shape[2]-num), size=X_train.shape[0], replace=True)
        res_index_col = (col_index.transpose()+choice_res.transpose()).transpose()

        choice_random_matrix = np.random.choice(range(self.Random_Matrix.shape[2]-num), size=X_train.shape[0], replace=True)
        random_matrix_index_col = (col_index.transpose()+choice_random_matrix.transpose()).transpose()
        snr_len = len(self.snrs)

        if low_snr == False:
            random_matrix_index_row = (Y_train*snr_len + (Z_train+20)/2).reshape([Y_train.shape[0],1]).cpu().numpy()
        else:
            #random_matrix_index_row = (Y_train*20).cpu().numpy().reshape([X_train.shape[0],1])
            random_matrix_index_row = (Y_train*snr_len).cpu().numpy().reshape([X_train.shape[0],1]) + np.random.randint(np.zeros(X_train.shape[0]),((Z_train+20)/2).cpu().numpy()+1,[X_train.shape[0],1])
        ones = np.ones([1, num])
        random_matrix_index_row = (random_matrix_index_row*ones).astype(int)
        res[row_index, :, res_index_col] = self.Random_Matrix[random_matrix_index_row, :, random_matrix_index_col]
        return res
    
    def discrete_random_mixing(self, X_train, Y_train, Z_train, p, low_snr=True):
        num = int(X_train.shape[2] * p)
        res = X_train.clone()
        snr_len = len(self.snrs)
        row_index = np.array([[i for i in range(X_train.shape[0])] for _ in range(num)]).transpose()
        res_index_col = np.random.choice(range(X_train.shape[2]), size=[X_train.shape[0],num], replace=True)


        random_matrix_index_col = np.random.choice(range(self.Random_Matrix.shape[2]), size=[X_train.shape[0],num], replace=True)
        if low_snr == False:
            random_matrix_index_row = (Y_train*snr_len + (Z_train+20)/2).reshape([Y_train.shape[0],1]).cpu().numpy()
        else:
            random_matrix_index_row = ((Y_train*snr_len).cpu().numpy().reshape([X_train.shape[0],1]) + np.random.randint(np.zeros(X_train.shape[0]),((Z_train+20)/2).cpu().numpy()+1,[X_train.shape[0],1]))
        ones = np.ones([1, num])
        random_matrix_index_row = (random_matrix_index_row*ones).astype(int)
        res[row_index, :, res_index_col] = self.Random_Matrix[random_matrix_index_row, :, random_matrix_index_col]
        return res

    def train(self, train_loader):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        iter_lr = AverageMeter('Iter lr')
        if (self.loss_name == "loss_CE_and_MSE") | (self.loss_name == "MixContrastiveLoss"):
            iter_loss_CE = AverageMeter('Iter loss_CE')
            iter_loss_MSE = AverageMeter('Iter loss_MSE')
        stime = time.time()
        for batch_idx, (X, Y, Z) in enumerate(data_loader):
            X, Y = X.to(self.device), Y.to(self.device)

            if self.mask_flag & self.model.training:
                X = self.zero_mask(X)
            if self.random_mix_flag & self.model.training:
                X = self.discrete_random_mixing(X,Y,Z,p=self.p,low_snr=True)

            if self.model_name == "TransformerMultiLoss":
                if self.model.training:
                    Y_soft = self.model(X,X)
                else:
                    Y_soft = self.model.predict(X)
            else:
                Y_soft = self.model(X)

            
            if (self.loss_name == "loss_CE_and_MSE") | (self.loss_name == "MixContrastiveLoss"):
                if self.model.training:
                    loss, Y_pred, loss_CE, loss_MSE = self.criterion(Y_soft, Y, X)
                    iter_loss_CE.update(loss_CE.item(), X.shape[0])
                    iter_loss_MSE.update(loss_MSE.item(), X.shape[0])
                else:
                    if self.model_name == "TransformerMultiLoss":
                        loss, Y_pred = self.criterion.calcCE(Y_soft, Y, X)
                    else:
                        loss, Y_pred, loss_CE, loss_MSE = self.criterion(Y_soft, Y, X)
                        iter_loss_CE.update(loss_CE.item(), X.shape[0])
                        iter_loss_MSE.update(loss_MSE.item(), X.shape[0])
            else:
                loss, Y_pred = self.criterion(Y_soft, Y, X)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            acc_pred = (Y_pred == Y).sum()
            acc_total = Y.numel()
            iter_acc.update(acc_pred/acc_total, acc_total)
            iter_loss.update(loss)
            iter_lr.update(self.optimizer.state_dict()['param_groups'][0]['lr'])
        
        if self.model.training:
            self.scheduler.step(self.scheduler_val_loss)
        else:
            self.scheduler_val_loss = iter_loss.avg
        ftime = time.time()

        if (self.loss_name == "loss_CE_and_MSE") | (self.loss_name == "MixContrastiveLoss"):
            if self.model.training:
                logger.info(f'Train | '
                            f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'loss_CE: {iter_loss_CE.avg:.3e} | '
                            f'loss_MSE: {iter_loss_MSE.avg:.3e} | '
                            f'Acc: {iter_acc.avg:.4f} | '
                            f'Counter: {self.counter} | '
                            f'time: {ftime-stime:.3f}')
            else:
                logger.info(f'Valid | '
                            f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'loss_CE: {iter_loss_CE.avg:.3e} | '
                            f'loss_MSE: {iter_loss_MSE.avg:.3e} | '
                            f'Acc: {iter_acc.avg:.4f} | '
                            f'time: {ftime-stime:.3f}')
        else:
            if self.model.training:
                logger.info(f'Train | '
                            f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'Acc: {iter_acc.avg:.3f} | '
                            f'Lr: {iter_lr.avg:.4e} | '
                            f'Counter: {self.counter} | '
                            f'time: {ftime-stime:.3f}')
            else:
                logger.info(f'Valid | '
                            f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'Acc: {iter_acc.avg:.4f} | '
                            f'time: {ftime-stime:.3f}')

        return iter_loss.avg.item(), iter_acc.avg.item()

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _loop_postprocessing(self, acc):
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }
        if self.best_acc is None:
            self.best_acc = acc
            state['best_acc'] = self.best_acc
            self._save(state, name=f"best_acc.pth")
        elif acc < self.best_acc + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.best_acc = acc
            state['best_acc'] = self.best_acc
            self._save(state, name=f"best_acc.pth")
            self.counter = 0


class Tester:
    def __init__(self, model, device, criterion, classes, snrs, loss_name = None, model_name = None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.classes = classes
        self.snrs = snrs
        self.conf = torch.zeros(classes, classes)
        self.conf_snr = torch.zeros(len(snrs), classes, classes)
        self.acc_pred_snr = torch.zeros(len(snrs))
        self.acc_total_snr = torch.zeros(len(snrs))
        self.acc_snr = torch.zeros(len(snrs))
        self.loss_name = loss_name
        self.model_name = model_name

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, acc = self._iteration(test_loader)

        # 混淆矩阵

        if verbose:
            logger.info(f'Test | '
                        f'loss: {loss:.3e} | '
                        f'Acc: {acc:.3f}')
        return loss, acc, self.conf, self.conf_snr, self.acc_snr

    def get_iq_framed(self, X, L=32, R=16):
        # [2, 1024]
        F=int((X.shape[2]-L)/R+1)
        Y = torch.zeros([F, X.shape[0], 2*L]).to(self.device)
        i = 0
        for idx in range(0, X.shape[-1]-L+1, R):
            Y[i, :, :] = X[:, :, idx:idx+L].reshape([1,X.shape[0], 2*L])
            i = i+1
            #Y.append(X[:, :, idx:idx+L].reshape([1,X.shape[0], 2*L]))  # (2, L=32)
        #Y = np.vstack(Y)  # (F, 2L) = (63, 64)  F=(1024-L)/R+1
        Y = Y.permute(1,0,2)
        return Y

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        stime = time.time()
        for batch_idx, (X, Y, Z) in enumerate(data_loader):
            X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
            
            if self.model_name == "TransformerMultiLoss":
                Y_soft = self.model.predict(X)
            else:
                Y_soft = self.model(X)
            

            if (self.loss_name == "loss_CE_and_MSE") | (self.loss_name == "MixContrastiveLoss"):
                if self.model_name == "TransformerMultiLoss":
                    loss, Y_pred = self.criterion.calcCE(Y_soft, Y, X)
                else:
                    loss, Y_pred, loss_ce, loss_mse = self.criterion(Y_soft, Y, X)
            else:
                loss, Y_pred = self.criterion(Y_soft, Y, X)

            acc_pred = (Y_pred == Y).sum()
            acc_total = Y.numel()
            for i in range(Y.shape[0]):
                self.conf[Y[i], Y_pred[i]] += 1
                idx = self.snrs.index(Z[i])
                self.conf_snr[idx, Y[i], Y_pred[i]] += 1
                self.acc_pred_snr[idx] += (Y[i] == Y_pred[i]).cpu()
                self.acc_total_snr[idx] += 1

            iter_acc.update(acc_pred / acc_total, acc_total)
            iter_loss.update(loss)
        for i in range(self.classes):
            self.conf[i, :] /= torch.sum(self.conf[i, :])
        for j in range(len(self.snrs)):
            self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]
            for i in range(self.classes):
                self.conf_snr[j, i, :] /= torch.sum(self.conf_snr[j, i, :])

        ftime = time.time()
        logger.info(f'Test | '
                    f'loss: {iter_loss.avg:.3e} | '
                    f'Acc: {iter_acc.avg:.4f} | '
                    f'time: {ftime-stime:.3f}')
        return iter_loss.avg.item(), iter_acc.avg.item()



