import gmi
import torch
from model import initialize_diffusion_model
from dataset import initialize_datasets_and_dataloaders

import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize diffusion model
image_shape = (3, 28, 28)
diffusion_model = initialize_diffusion_model(image_shape).to(device)
assert isinstance(diffusion_model, gmi.diffusion.DiffusionModel)

# load model
diffusion_model.load_state_dict(torch.load('./weights/diffusion_model.pth'))

# set the diffusion model to evaluation mode
diffusion_model.eval()

# initialize datasets and dataloaders
medmnist_train_dataset, \
    medmnist_val_dataset, \
    medmnist_test_dataset, \
    medmnist_train_dataloader, \
    medmnist_val_dataloader, \
    medmnist_test_dataloader, \
    image_shape = initialize_datasets_and_dataloaders('BloodMNIST', 1)

# intialize x_0
x_0, y  = next(iter(medmnist_train_dataloader))
x_0 = x_0.to(device)
y = y.to(device)

# initialize the reverse process at x_T
T = torch.tensor([10000.0],dtype=torch.float32).reshape(1, 1).to(device)
x_T = diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0, T)

# predict x_0
x_0_pred = diffusion_model.predict_x_0(x_T, T, y)

# plot three images, x_0, x_T, x_0_pred
fig, ax = plt.subplots(1, 3)
ax[0].imshow(x_0[0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[0].axis('off')
ax[0].set_title('x_0')
ax[1].imshow(x_T[0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[1].axis('off')
ax[1].set_title('x_T')
ax[2].imshow(x_0_pred[0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[2].axis('off')
ax[2].set_title('x_0_pred')
plt.savefig('figures/x_0_pred.png')


# redo it with no information from the original image for unconditional sampling
x_T = diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0*0, T)
y = torch.ones(1, 1).to(device)*-1

# sample the reverse process
timesteps = (torch.linspace(1.0, 0.0, 100).to(device)**4.0)*T[0,0]
x_t_all = diffusion_model.sample_reverse_process(x_T, timesteps, sampler='euler', return_all=True, y=y, verbose=True)


# visualize with an animation

fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(x_t_all[0][0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax.set_title('Reverse Diffusion t= {}'.format(timesteps[0]))
def update(i):
    im.set_array(x_t_all[i][0,:].detach().cpu().permute(1,2,0).numpy())
    ax.set_title('Reverse Diffusion t= {}'.format(timesteps[i]))
    return im,

ani = animation.FuncAnimation(fig, update, frames=range(len(timesteps)), blit=True)
writer = animation.writers['ffmpeg'](fps=5)
ani.save('figures/reverse_diffusion.mp4', writer=writer)





# re do the x_0 plot but with the reverse diffusion result
fig, ax = plt.subplots(1, 4)
ax[0].imshow(x_0[0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[0].axis('off')
ax[0].set_title('x_0')
ax[1].imshow(x_T[0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[1].axis('off')
ax[1].set_title('x_T')
ax[2].imshow(x_0_pred[0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[2].axis('off')
ax[2].set_title('E[x_0|x_T]')
ax[3].imshow(x_t_all[-1][0,:].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
ax[3].axis('off')
ax[3].set_title('x_0 | x_T')
plt.savefig('figures/x_0_diffusion_sample.png')
plt.show()





# initialize datasets and dataloaders
medmnist_train_dataset, \
    medmnist_val_dataset, \
    medmnist_test_dataset, \
    medmnist_train_dataloader, \
    medmnist_val_dataloader, \
    medmnist_test_dataloader, \
    image_shape = initialize_datasets_and_dataloaders('BloodMNIST', 9)

# Sample 9 images from the BloodMNIST dataset
batch_size = 9
x_0_batch, _ = next(iter(medmnist_train_dataloader))
x_0_batch = x_0_batch[:batch_size].to(device)

# Run reverse diffusion for the sampled batch
T_batch = torch.tensor([10000.0], dtype=torch.float32).reshape(1, 1).repeat(batch_size, 1).to(device)
x_T_batch = diffusion_model.forward_SDE.sample_x_t_given_x_0(x_0_batch, T_batch)
y_batch = torch.ones(batch_size, 1).to(device) * -1  # Unconditional sampling

# Sample the reverse process
timesteps_batch = (torch.linspace(1.0, 0.0, 100).to(device) ** 4.0) * T_batch[0, 0]
x_t_all_batch = diffusion_model.sample_reverse_process(x_T_batch, timesteps_batch, sampler='euler', return_all=False, y=y_batch, verbose=False)

# Plot the grid
fig, axes = plt.subplots(3, 7, figsize=(14, 6))
for i in range(3):
    for j in range(7):
        ax = axes[i, j]
        ax.axis('off')
        
        if j < 3:
            # Plot samples from BloodMNIST dataset
            ax.imshow(x_0_batch[i * 3 + j].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        elif j > 3:
            # Plot reverse diffusion samples
            ax.imshow(x_t_all_batch[i * 3 + (j - 4)].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')

# Add titles
fig.text(0.25, 0.95, 'Samples from BloodMNIST Dataset', ha='center', fontsize=16)
fig.text(0.75, 0.95, 'Reverse Diffusion Samples', ha='center', fontsize=16)

plt.savefig('figures/diffusion_grid.png')
plt.show()