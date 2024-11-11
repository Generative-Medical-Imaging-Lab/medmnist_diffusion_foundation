import torch
import argparse
from dataset import medmnist_mapping, initialize_datasets, create_dataloader
from model import initialize_diffusion_model
from train import train_medmnist_diffusion_foundation_model
from sample import generate_samples

def train_and_sample_model(dataset_name, device):
    num_epochs = 100
    num_iterations = 100
    training_batch_size = 256
    sampling_batch_size = 9
    num_steps = 300
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu')

    print(f"\n--- Processing {dataset_name} on device {device} ---")

    # Initialize datasets
    train_dataset, val_dataset, test_dataset, image_shape, y_dim, model_path = initialize_datasets(dataset_name)

    # Create data loaders with appropriate batch sizes
    train_dataloader = create_dataloader(train_dataset, batch_size=training_batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size=training_batch_size)
    test_dataloader = create_dataloader(test_dataset, batch_size=sampling_batch_size, shuffle=False)        

    # Model save path
    model_save_path = f'weights/{dataset_name}_diffusion_model.pth'
    
    try:
        # Load the model if it already exists
        diffusion_model = initialize_diffusion_model(image_shape, y_dim, device=device)
        diffusion_model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Model loaded from {model_save_path}")
    except FileNotFoundError:
        # Train the model if it doesn't exist
        print(f"Training diffusion model for {dataset_name}...")
        diffusion_model = initialize_diffusion_model(image_shape, y_dim).to(device)

        # Train the model
        diffusion_model = train_medmnist_diffusion_foundation_model(
            diffusion_model=diffusion_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_path=model_path,
            batch_size=training_batch_size,
            num_epochs=num_epochs,
            num_iterations=num_iterations,
            device=device,
            wandb_project=None
        )

        # Save the trained model weights
        torch.save(diffusion_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Generate samples using the trained model
    print(f"Generating samples for {dataset_name}...")
    output_path = f'figures/{dataset_name}_sample.png'
    generate_samples(diffusion_model, test_dataloader=test_dataloader, num_steps=num_steps, output_path=output_path, device=device)
    print(f"Sample images saved to {output_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train and sample diffusion model for a specific MedMNIST dataset.")
    # parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to process.")
    # parser.add_argument('--device', type=int, default=0, help="CUDA device index to use (e.g., 0). Use -1 for CPU.")
    
    # args = parser.parse_args()
    # train_and_sample_model(args.dataset_name, args.device)



    for dataset_name in medmnist_mapping.keys():
        train_and_sample_model(dataset_name, 0)
