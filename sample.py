import gmi
import torch
import matplotlib.pyplot as plt

def generate_samples(diffusion_model, test_dataloader, num_steps=100, output_path='figures/diffusion_grid.png', device='cuda' if torch.cuda.is_available() else 'cpu'):
    assert isinstance(diffusion_model, gmi.diffusion.DiffusionModel)
    
    # T = 10000.0
    log10_T = -4.0
    log10_t0 = -4.0
    T = 10**log10_T
    
    # Sample a batch from the test dataloader
    x_0, y = next(iter(test_dataloader))
    x_0 = x_0.to(device)
    y = y.to(device)

    # Initialize reverse diffusion process at x_T
    T = torch.tensor([T], dtype=torch.float32).reshape(1, 1).repeat(x_0.shape[0], 1).to(device)
    # x_T = diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0 * 0, T)
    x_T = diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0 , T)

    # Define y as negative ones for unconditional sampling
    y = y*0 
    
    # Define timesteps for reverse diffusion
    timesteps = torch.logspace(log10_T, log10_t0, num_steps).to(device)
    x_t_all = diffusion_model.sample_reverse_process(x_T, timesteps, sampler='euler', return_all=False, y=y, verbose=True)

    # Plot and save samples in a 3x7 grid
    fig, axes = plt.subplots(3, 7, figsize=(14, 6))

    # Loop over grid positions
    for i in range(3):
        for j in range(7):
            ax = axes[i, j]
            ax.axis('off')
            
            if j < 3:
                # Original samples
                ax.imshow(x_0[i * 3 + j].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
            elif j > 3:
                # Diffusion samples
                ax.imshow(x_t_all[i * 3 + (j - 4)].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')

    # Add centered titles for the original and diffusion sample grids
    dataset_title = test_dataloader.dataset.__class__.__name__.replace("Dataset", "")
    fig.text(0.25, 0.95, f'{dataset_title} Samples', ha='center', fontsize=16)
    fig.text(0.75, 0.95, f'{dataset_title}Diffusion Samples', ha='center', fontsize=16)

    plt.savefig(output_path)
    # plt.show()
