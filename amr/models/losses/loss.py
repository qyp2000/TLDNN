import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ["loss_CE", "loss_CE_and_MSE", "loss_Focal", "MixContrastiveLoss", "mixTripletLoss"]


class loss_CE(nn.Module):
    def __init__(self):
        super(loss_CE, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label, ori):
        loss = self.ce(pred, label)
        Y_pred = torch.argmax(pred, 1)
        return loss, Y_pred

    def __call__(self, pred, label, ori=None):
        return self.forward(pred, label, ori)


class loss_CE_and_MSE(nn.Module):
    def __init__(self, alpha=0.1):
        super(loss_CE_and_MSE, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, label, ori):  # pred(out1,out2)
        loss1 = self.ce(pred[1], label)
        loss2 = self.mse(pred[0], ori)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        Y_pred = torch.argmax(pred[1], 1)
        return loss, Y_pred, loss1, loss2
    
    def calcCE(self, pred, label, ori):
        loss = self.ce(pred, label)
        Y_pred = torch.argmax(pred, 1)
        return loss, Y_pred

    def __call__(self, pred, label, ori=None):
        return self.forward(pred, label, ori)


class loss_Focal(nn.Module):
    def __init__(self, class_num = 11, alpha=None, gamma=2, size_average=True):
        super(loss_Focal, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets,ori):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        
        Y_pred = torch.argmax(inputs, 1)
        return loss,Y_pred


class MixContrastiveLoss():
    def __init__(self, margin = 0.05, gpu = 0, lam = 10):
        super(MixContrastiveLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(margin=margin,gpu=gpu)
        self.softmax_loss = torch.nn.CrossEntropyLoss()
        self.lam=lam
    def __call__(self, pred: torch.Tensor, labels: torch.Tensor, ori) -> torch.Tensor:
        loss_contrastive = self.contrastive_loss(pred[0], labels)
        loss_ce = self.softmax_loss(pred[1],labels)
        loss = self.lam*loss_contrastive+loss_ce
        Y_pred = torch.argmax(pred[1], 1)
        return loss, Y_pred, loss_ce, loss_contrastive


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0,gpu=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gpu = gpu
    def _pairwise_distances(self, embeddings):
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = torch.max(distances, torch.tensor([0.0]).cuda(self.gpu))
        return distances

    def _get_Contrastive_mask(self, labels):
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.shape[0]).cuda(self.gpu)
        indices_not_equal = torch.tensor([1.0]).cuda(self.gpu)-indices_equal

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1)).float()
        label_not_equal = torch.tensor([1.0]).cuda(self.gpu) - label_equal

        # Combine the two masks
        label_equal = torch.mul(indices_not_equal, label_equal)
        label_not_equal = torch.mul(indices_not_equal, label_not_equal)
        return label_equal, label_not_equal



    def batch_all_contrastive_loss(self, labels, embeddings, loss_all = True):       
        pairwise_dist = self._pairwise_distances(embeddings)
        label_equal, label_not_equal = self._get_Contrastive_mask(labels)

        contrastive_loss_equal = torch.mul(label_equal, pairwise_dist)
        contrastive_loss_not_equal = torch.mul(label_not_equal, self.margin - pairwise_dist)
        contrastive_loss_not_equal = torch.max(contrastive_loss_not_equal, torch.tensor([0.0]).cuda(self.gpu))
        if loss_all:
            contrastive_loss = (torch.sum(contrastive_loss_equal) + torch.sum(contrastive_loss_not_equal))/labels.shape[0]/(labels.shape[0]-1)
        else:
            contrastive_loss = torch.sum(contrastive_loss_equal)/torch.sum(label_equal)

        return contrastive_loss

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.batch_all_contrastive_loss(labels, embeddings)
        return loss



class mixTripletLoss():
    def __init__(self, margin = 0.05, gpu = 0, lam = 5):
        super(mixTripletLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=margin,gpu=gpu)
        self.softmax_loss = torch.nn.CrossEntropyLoss()
        self.lam=lam
    def __call__(self, pred: torch.Tensor, labels: torch.Tensor, ori, squared=False) -> torch.Tensor:
        loss_triplet = self.triplet_loss(pred[0], labels, squared)
        loss_ce = self.softmax_loss(pred[1],labels)
        loss = self.lam*loss_triplet+loss_ce
        Y_pred = torch.argmax(pred[1], 1)
        return loss, Y_pred, loss_ce, loss_triplet

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0,gpu=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.gpu = gpu
    def _pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = torch.max(distances, torch.tensor([0.0]).cuda(self.gpu))

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = (torch.eq(distances, 0.0)).float()
            distances = distances + mask * 1e-16

            distances = torch.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (torch.sub(1.0, mask))

        return distances

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.shape[0]).cuda(self.gpu)
        indices_not_equal = torch.tensor([1.0]).cuda(self.gpu)-indices_equal
        i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
        i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
        j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

        distinct_indices = torch.mul(torch.mul(i_not_equal_j, i_not_equal_k), j_not_equal_k)


        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1)).float()
        i_equal_j = torch.unsqueeze(label_equal, 2)
        i_equal_k = torch.unsqueeze(label_equal, 1)
        valid_labels = torch.mul(i_equal_j, torch.tensor([1.0]).cuda(self.gpu)-i_equal_k)

        # Combine the two masks
        mask = torch.mul(distinct_indices, valid_labels)
        return mask


    def batch_all_triplet_loss(self, labels, embeddings, squared=False):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        margin = self.margin
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels)
        mask = mask.float()
        triplet_loss = torch.mul(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = torch.max(triplet_loss, torch.tensor([0.0]).cuda(self.gpu))

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_positive_triplets = torch.sum(valid_triplets)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, squared=False) -> torch.Tensor:
        loss = self.batch_all_triplet_loss(labels, embeddings, squared)
        return loss
