"""
Class for training a model for a single epoch (i.e., one run over the dataset).
"""

import time
import torch
from modeling.model_runner import ModelRunner


class TrainEpoch(ModelRunner):
    """
    Base class for running one epoch of model training.
    """
    def __init__(self, model, model_args, multi_process_args, loss_args, logger, dataloader, opt, stop_checker,
                 train_hist):
        """
        :param model: model to train
        :param model_args: arguments passed into model
        :param multi_process_args: arguments related to multi-processing
        :param loss_args: arguments related to loss function
        :param logger: module used for logging progress
        :param dataloader: data to use
        :param opt: optimizer
        :param stop_checker: module for checking stopping criteria
        :param train_hist: module for storing training progress stats
        """
        super().__init__(model, model_args, multi_process_args, loss_args, logger)
        self.dataloader = dataloader
        self.opt = opt
        self.stop_checker = stop_checker
        self.train_hist = train_hist

    def run_train_epoch(self):
        self.logger.info(f"Starting epoch {self.train_hist.epoch}")
        self.model.train()
        
        # Add progress reporting
        total_batches = len(self.dataloader)
        log_interval = max(1, total_batches // 10)  # Log approximately 10 times per epoch
        
        # Track time for speed reporting
        start_time = time.time()
        last_time = start_time
        samples_processed = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Process batch
            self.model.zero_grad()
            self._train_step(batch)
            self.opt.step()
            
            # Update tracking variables
            batch_size = len(batch)
            samples_processed += batch_size
            
            # Display progress periodically
            if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
                # Get current loss
                current_loss = self.train_hist.train_loss_steps[self.train_hist.epoch][-1]
                if isinstance(current_loss, torch.Tensor):
                    current_loss = current_loss.item()
                
                # Calculate speed
                current_time = time.time()
                elapsed = current_time - last_time
                examples_per_sec = (batch_size * log_interval) / elapsed if batch_idx > 0 else 0
                
                # Calculate ETA
                if batch_idx > 0:
                    progress = (batch_idx + 1) / total_batches
                    time_so_far = current_time - start_time
                    estimated_total = time_so_far / progress
                    eta_seconds = estimated_total - time_so_far
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_min}m {eta_sec}s"
                else:
                    eta_str = "calculating..."
                
                # Print progress
                print(f"Epoch {self.train_hist.epoch} | Batch {batch_idx+1}/{total_batches} | " 
                      f"Loss: {current_loss:.6f} | "
                      f"Speed: {examples_per_sec:.1f} ex/s | "
                      f"Progress: {100 * (batch_idx+1) / total_batches:.1f}% | "
                      f"ETA: {eta_str}")
                
                # Update timing reference
                last_time = current_time
            
            # Check if we should stop training
            stop_training = self.stop_checker.check_stop(self.train_hist.epoch,
                                                        self.train_hist.step,
                                                        self.train_hist.sample_count)
            if stop_training:
                break
        
        # Calculate total time for epoch
        epoch_time = time.time() - start_time
        epoch_min = int(epoch_time // 60)
        epoch_sec = int(epoch_time % 60)
                
        # Finalize epoch
        mean_epoch_loss = self.train_hist.increment_epoch()
        self.logger.info(f"Epoch {self.train_hist.epoch-1} completed in {epoch_min}m {epoch_sec}s | "
                         f"Mean loss: {mean_epoch_loss:.6f} | "
                         f"Samples: {samples_processed}")
        print(f"Epoch {self.train_hist.epoch-1} completed in {epoch_min}m {epoch_sec}s | "
              f"Mean loss: {mean_epoch_loss:.6f} | "
              f"Samples: {samples_processed}")

    def _train_step(self, batch):
        loss = self.model(batch)
        if self.multi_process_args.apply_data_parallel and self.multi_process_args.n_gpu > 1:
            # aggregate loss from multiple GPUS
            loss = self._mp_aggregate_loss(loss)
        loss.backward()
        self.train_hist.increment_step(loss.detach(), len(batch))
