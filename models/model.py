"""
Base class for neural network models.
"""

import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Base class for neural network model.
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = None
        self.nn_dict = None

    def _check_loss_fn_set(self):
        if self.loss_fn is None:
            print("ERROR: need to set model loss function before calling forward!")
            print("Exiting...")
            exit(1)

    def save(self, output_dir, fname_prefix):
        # save whole model
        file_path = os.path.join(output_dir, fname_prefix+".pt")
        state_dict = self.state_dict()
        torch.save(state_dict, file_path)
        # save sub nets within model
        for nn_name, net in self.nn_dict.items():
            file_path = os.path.join(output_dir, fname_prefix+f"_{nn_name}.pt")
            state_dict = net.state_dict()
            torch.save(state_dict, file_path)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def _prepare_inputs(self, inputs, prep_keys=None):
        """
        :param inputs: inputs to model to prepare for running on NN model
                       (i.e., put on the right device & convert to float)
        :param prep_keys: (iterable) keys of items to prepare for running through NN model
        :return: processed inputs
        """
        if prep_keys is None:
            prep_keys = set(inputs.keys())
            
        # Collect all tensors that need to be moved to device
        tensors_to_move = {}
        for k, v in inputs.items():
            if k in prep_keys and isinstance(v, torch.Tensor):
                tensors_to_move[k] = v
        
        # If we have tensors to move, move them all at once with a single CUDA operation
        if tensors_to_move:
            # Create non_blocking=True only for CUDA devices
            non_blocking = (self.device.type == 'cuda')
            for k, v in tensors_to_move.items():
                # Convert to float if needed and move to device in one operation
                if v.dtype != torch.float:
                    inputs[k] = v.float().to(device=self.device, non_blocking=non_blocking)
                else:
                    inputs[k] = v.to(device=self.device, non_blocking=non_blocking)
                    
        return inputs

    def _prepare_targets(self, targets):
        """
        :param targets: prediction targets to prepare to use as argument to loss function
                       (i.e., put on the right device & convert to float if needed)
        :return: processed targets
        """
        # Use non_blocking transfer for CUDA devices
        non_blocking = (self.device.type == 'cuda')
        
        # Handle typing and dimensionality in a single operation when possible
        if len(targets.shape) == 1:
            # For 1D targets, unsqueeze and convert to float in one step
            if targets.dtype == torch.double:
                return targets.float().unsqueeze(1).to(device=self.device, non_blocking=non_blocking)
            else:
                # Still need to convert to float for binary classification
                return targets.unsqueeze(1).float().to(device=self.device, non_blocking=non_blocking)
        else:
            # For 2D targets, just handle dtype
            if targets.dtype == torch.double:
                return targets.float().to(device=self.device, non_blocking=non_blocking)
            return targets.to(device=self.device, non_blocking=non_blocking)
