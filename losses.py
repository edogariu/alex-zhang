import pytorch_metric_learning.losses as pml
import torch
import torch.nn.functional as F

reconstruction_loss = F.mse_loss

_CONTRASTIVE_LOSS = pml.SupConLoss(temperature=0.1)

def contrastive_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Computes contrastive loss, where corresponding data points in `emb1` and `emb2` are assumed to come from the same source, and all other pairs between `emb1` and `emb2` come from different sources.
    This loss pushes the embeddings for corresponding data points together, and different data points apart
    
    for example, 
    `emb1 = [img1.RGB_structure, img2.RGB_structure, ...]` and `emb2 = [img1.DEPTH_structure, img2.DEPTH_structure, ...]`

    Parameters
    ----------
    emb1 : torch.Tensor
        tensor of shape `(B, emb_dim)` of embeddings of type 1
    emb2 : torch.Tensor
        tensor of shape `(B, emb_dim)` of embeddings of type 2

    Returns
    -------
    torch.Tensor
        loss
    """
    assert emb1.shape == emb2.shape, 'must be same shape'
    B = emb1.shape[0]
    labels = torch.arange(2 * B).to(emb1.device)  # makes "class labels" s.t. corresponding data points are in the same class
    labels[B:] -= B
    x = torch.cat((emb1, emb2), dim=0)
    loss = _CONTRASTIVE_LOSS(x, labels)
    return loss

def anticontrastive_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Computes anticontrastive loss, where corresponding data points in `emb1` and `emb2` are assumed to come from the same source, and all other pairs between `emb1` and `emb2` come from different sources.
    This loss tries to make corresponding embeddings uncorrelated, and does nothing across different data points
    
    for example, 
    `emb1 = [img1.RGB_appearance, img2.RGB_appearance, ...]` and `emb2 = [img1.RGB_structure, img2.RGB_structure, ...]`

    Parameters
    ----------
    emb1 : torch.Tensor
        tensor of shape `(B, emb_dim)` of embeddings of type 1
    emb2 : torch.Tensor
        tensor of shape `(B, emb_dim)` of embeddings of type 2

    Returns
    -------
    torch.Tensor
        loss
    """
    assert emb1.shape == emb2.shape, 'must be same shape'
    dots = (emb1 * emb2).sum(dim=-1)  # computes dot product between corresponding embeddings
    loss = (dots ** 2).mean()
    return loss

if __name__ == '__main__':
    emb1 = torch.tensor([[5, 1], [3, 4], [4, -1]]).float()
    emb2 = torch.tensor([[-1, 1], [3, 4], [4, -1]]).float()
    print(contrastive_loss(emb1, emb2))
    print(anticontrastive_loss(emb1, emb2))
