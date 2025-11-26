import torch
from torch import nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z):
        """
        Compute the NT-Xent loss for a batch of embeddings.

        Args:
        - z: Tensor of shape (2N, D) representing the concatenation of two sets of augmented embeddings (each of shape (N, D))

        Returns:
        - Loss value (scalar).
        """
        N = z.shape[0] // 2  # Batch size (// to have an int and not float)

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(z, z.T)  # (2N, 2N)

        # Create mask to exclude self-comparisons
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Compute log-softmax over all negatives
        logits = similarity_matrix
        log_prob = F.log_softmax(logits, dim=1)

        # Extract log-probabilities of positive pairs
        positives_idx = torch.cat(
            [torch.arange(N, 2 * N), torch.arange(N)]
        )  # Indices of positives in log_prob matrix
        loss = -log_prob[torch.arange(2 * N), positives_idx].mean()
    
        return loss