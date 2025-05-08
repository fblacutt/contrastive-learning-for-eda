"""
Loss functions for contrastive learning.
"""
import torch.nn as nn
import torch


class NCELoss(nn.Module):
    """
    Noise Contrastive Estimation Loss
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_v1, embeddings_v2, key_ids=None):
        """
        embeddings_v1: view 1 of learned representations (h) for each sample, potentially transformed
        embeddings_v2: view 2 of learned representations (h) for each sample, transformed
        key_ids (optional): ids denoting which samples should be considered the same vs different for
                 contrastive learning. If not provided, will assume each instance is the same only to
                 itself (i.e., embeddings_v1[i] and embeddings_v2[i] are pos pairs)
        """
        # Normalize embeddings for cosine similarity computation - more efficient approach
        embeddings_v1 = torch.nn.functional.normalize(embeddings_v1, p=2, dim=1)
        embeddings_v2 = torch.nn.functional.normalize(embeddings_v2, p=2, dim=1)
        
        # Compute similarity matrix directly using normalized embeddings
        sim_matrix = torch.mm(embeddings_v1, embeddings_v2.transpose(0, 1)) / self.temperature
        sim_matrix_exp = torch.exp(sim_matrix)
        
        # Handle key_ids for identifying positive pairs
        if key_ids is None:
            # Diagonal mask is more efficient when each item is only positive with itself
            batch_size = embeddings_v1.size(0)
            pos_mask = torch.eye(batch_size, device=embeddings_v1.device, dtype=torch.bool)
        else:
            # Create mask for more complex positive pair relationships
            key_ids1, key_ids2 = torch.meshgrid(key_ids, key_ids, indexing='ij')
            pos_mask = key_ids1 == key_ids2

        # Compute loss for both views in a more efficient way
        # (1) view 1 → view 2 direction
        row_sum = torch.sum(sim_matrix_exp, dim=1, keepdim=True)
        sim_matrix_row_normalize = sim_matrix_exp / row_sum
        view1_loss = -torch.mean(torch.log(sim_matrix_row_normalize[pos_mask]))
        
        # (2) view 2 → view 1 direction
        col_sum = torch.sum(sim_matrix_exp, dim=0, keepdim=True)
        sim_matrix_col_normalize = sim_matrix_exp / col_sum
        view2_loss = -torch.mean(torch.log(sim_matrix_col_normalize[pos_mask]))
        
        loss = (view1_loss + view2_loss) / 2
        return loss
