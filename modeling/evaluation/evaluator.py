"""
Module for evaluating model.
"""
import time

import torch
from torch.utils.data.dataloader import DataLoader

from modeling.evaluation.eval_results import EvalResults
from modeling.model_runner import ModelRunner


class Evaluator(ModelRunner):
    """
    Base class for evaluating model
    """
    def __init__(self, model, model_args, multi_process_args, loss_args, logger, eval_dataset, eval_args):
        """
        :param model: model to use
        :param model_args: arguments passed into model
        :param multi_process_args: arguments related to multi-processing
        :param loss_args: arguments related to loss function
        :param logger: module used for logging progress
        :param eval_dataset: dataset ot use for model evaluation
        :param eval_args: arguments related to evaluation
        """
        super().__init__(model, model_args, multi_process_args, loss_args, logger)
        self.eval_dataset = eval_dataset
        self.eval_args = eval_args

    def evaluate(self):
        self.logger.info("Starting EVALUATE")
        eval_start_time = time.time()
        
        # Run evaluation
        eval_dataloader = self._get_eval_dataloader()
        eval_results = self._eval_epoch(eval_dataloader)
        
        # Format metrics for display
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in eval_results.metrics.items()])
        
        # Calculate total evaluation time
        eval_time = time.time() - eval_start_time
        minutes = int(eval_time // 60)
        seconds = int(eval_time % 60)
        time_str = f"{minutes}m {seconds}s"
        
        # Add time info to results
        eval_results.add_time_info(eval_time)
        
        # Format and display summary
        eval_summary = f"EVALUATION COMPLETED in {time_str}:\n{metrics_str}"
        self.logger.info(eval_summary)
        print("\n" + eval_summary + "\n")
        
        return eval_results

    def _eval_epoch(self, eval_dataloader):
        store_preds = self.eval_args.save_preds or self.eval_args.metrics is not None
        num_batches = len(eval_dataloader)
        num_samples = len(self.eval_dataset)
        eval_results = EvalResults(num_batches,
                                   num_samples,
                                   store_preds,
                                   self.model.output_dim,
                                   self.eval_dataset.y_dim,
                                   self.eval_dataset.y_var_type)
        self.model.eval()
        
        # Add progress reporting
        log_interval = max(1, num_batches // 5)  # Show progress ~5 times
        start_time = time.time()
        
        print(f"Evaluating on {num_samples} samples...")
        
        for step, batch in enumerate(eval_dataloader):
            # Process batch
            loss, preds = self._eval_step(batch)
            eval_results.add_step_result(step, loss, preds, batch)
            
            # Display progress periodically
            if step % log_interval == 0 or step == num_batches - 1:
                progress_pct = 100 * (step + 1) / num_batches
                elapsed = time.time() - start_time
                
                # Calculate ETA
                if step > 0:
                    eta_seconds = (elapsed / (step + 1)) * (num_batches - step - 1)
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_min}m {eta_sec}s"
                else:
                    eta_str = "calculating..."
                
                print(f"Evaluation: {step+1}/{num_batches} batches | "
                      f"Progress: {progress_pct:.1f}% | ETA: {eta_str}")
        
        # Finalize evaluation
        eval_results.compute_mean_loss()
        if self.eval_args.metrics is not None:
            eval_results.compute_metrics(self.eval_args.metrics, normalize_preds=True)
        
        return eval_results

    def _eval_step(self, batch):
        with torch.no_grad():
            loss, preds = self.model(batch, return_preds=True)
            if self.multi_process_args.apply_data_parallel and self.multi_process_args.n_gpu > 1:
                loss = loss.mean()
        return loss, preds

    def _get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_args.batch_size,
            shuffle=False,
            num_workers=self.multi_process_args.num_workers,
            pin_memory=True,
            drop_last=False
        )
