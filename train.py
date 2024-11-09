import gmi
import medmnist
import torch
from torch import nn
from torchvision import transforms
from model import initialize_diffusion_model
from dataset import initialize_datasets_and_dataloaders

import lpips




def train_medmnist_diffusion_foundation_model(
                                    dataset_name,
                                    batch_size=16,
                                    num_epochs=100,
                                    num_iterations=100,
                                    num_iterations_val=10,                           
                                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    medmnist_train_dataset, \
    medmnist_val_dataset, \
    medmnist_test_dataset, \
    medmnist_train_dataloader, \
    medmnist_val_dataloader, \
    medmnist_test_dataloader, \
    image_shape = initialize_datasets_and_dataloaders(dataset_name, batch_size)

    diffusion_model = initialize_diffusion_model(image_shape)

    def time_sampler(batch_size):
        mean_log_t = -3.0
        std_log_t = 2.0
        log_t = torch.randn(batch_size, 1).to(device)*std_log_t + mean_log_t
        t = torch.exp(log_t)
        return t
    
    class DiffusionLossClosure(nn.Module):
        def __init__(self, diffusion_model, time_sampler):
            super(DiffusionLossClosure, self).__init__()
            self.diffusion_model = diffusion_model
            self.time_sampler = time_sampler

            self.loss_fn = torch.nn.MSELoss()

            # self.lpips = lpips.LPIPS(net='alex').to(device)
            # n = 2 + 16+32+64
            # def loss_fn(x_0, x_0_pred):
            #     x_0 = torch.nn.functional.pad(x_0, (n, n, n, n), mode='constant', value=0.0)
            #     x_0_pred = torch.nn.functional.pad(x_0_pred, (n, n, n, n), mode='constant', value=0.0)
            #     lpips_loss = self.lpips(x_0, x_0_pred)
            #     return torch.mean(lpips_loss)
            
            # self.loss_fn = loss_fn

        def forward(self, x_0, y):
            assert isinstance(x_0, torch.Tensor)
            assert isinstance(y, torch.Tensor) or y is None
            assert isinstance(self.diffusion_model, gmi.diffusion.DiffusionModel)
            assert isinstance(self.diffusion_model.forward_SDE, 
                              gmi.sde.StochasticDifferentialEquation)
            
            # set the second half of y to -1, for unconditional training
            if y is not None:
                y[y.shape[0]//2:] = -1

            t = self.time_sampler(x_0.shape[0])
            x_t = self.diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0, t)
            x_0_pred = self.diffusion_model.predict_x_0(x_t, t, y)
            
            loss = self.loss_fn(x_0_pred, x_0)

            return loss
        
    diffusion_loss_closure = DiffusionLossClosure(diffusion_model, time_sampler)

    gmi.train(  
        medmnist_train_dataloader, 
            diffusion_loss_closure, 
            num_epochs=num_epochs, 
            num_iterations=num_iterations,
            optimizer=None,
            lr=1e-3, 
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            validation_loader=medmnist_val_dataloader, 
            num_iterations_val=num_iterations_val,
            verbose=True,
            very_verbose=True)
    
    return diffusion_model


diffusion_model = train_medmnist_diffusion_foundation_model(
                                    dataset_name='BloodMNIST',                           
                                    device='cuda' if torch.cuda.is_available() else 'cpu')

# Save the model
torch.save(diffusion_model.state_dict(), 'diffusion_model.pth')