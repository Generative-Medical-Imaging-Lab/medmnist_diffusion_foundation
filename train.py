import gmi
import torch
from torch import nn
import wandb

def train_medmnist_diffusion_foundation_model(diffusion_model, train_dataloader, val_dataloader, model_path, batch_size=32, num_epochs=100, num_iterations=100, num_iterations_val=10, device='cuda' if torch.cuda.is_available() else 'cpu', wandb_project=None):
    # Initialize WandB logging if provided
    if wandb_project:
        wandb.init(project=wandb_project)
        wandb.config.update({"batch_size": batch_size, "num_epochs": num_epochs})

    def time_sampler(batch_size):
        mean_log_t = -1.0
        std_log_t = 3.0
        log_t = torch.randn(batch_size, 1).to(device) * std_log_t + mean_log_t
        return torch.exp(log_t)

    class DiffusionLossClosure(nn.Module):
        def __init__(self, diffusion_model, time_sampler):
            super(DiffusionLossClosure, self).__init__()
            self.diffusion_model = diffusion_model
            self.time_sampler = time_sampler

            def loss_fn(x_0_pred, x_0, t):
                weights = 1 / t.reshape([t.shape[0]] + [1] * (len(x_0.shape) - 1))
                return torch.mean(weights * (x_0_pred - x_0) ** 2)

            self.loss_fn = loss_fn

        def forward(self, x_0, y):
            t = self.time_sampler(x_0.shape[0])
            x_t = self.diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0, t)
            x_0_pred = self.diffusion_model.predict_x_0(x_t, t, y)
            return self.loss_fn(x_0_pred, x_0, t)

    diffusion_loss_closure = DiffusionLossClosure(diffusion_model, time_sampler)

    # Train the model using the provided data loaders
    gmi.train(train_dataloader, diffusion_loss_closure, num_epochs=num_epochs, num_iterations=num_iterations, optimizer=None, lr=1e-3, device=device, validation_loader=val_dataloader, num_iterations_val=num_iterations_val, verbose=True, very_verbose=False)

    if wandb_project:
        wandb.finish()

    return diffusion_model
