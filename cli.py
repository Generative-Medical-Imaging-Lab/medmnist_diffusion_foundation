import argparse
import torch
from model import initialize_diffusion_model
from sample import generate_samples
from train import train_medmnist_diffusion_foundation_model
from dataset import medmnist_mapping

def main():
    parser = argparse.ArgumentParser(description="Train or sample with the diffusion model.")
    parser.add_argument('--task', type=str, choices=['train', 'sample'], required=True, help='Task to perform')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., BloodMNIST)')
    parser.add_argument('--batch_size', type=int, default=9, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of reverse diffusion steps')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name for logging')
    parser.add_argument('--output', type=str, default='figures/diffusion_grid.png', help='Output path for samples')
    
    args = parser.parse_args()

    if args.task == 'train':
        train_medmnist_diffusion_foundation_model(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            wandb_project=args.wandb_project
        )
    elif args.task == 'sample':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, image_shape, model_path = medmnist_mapping[args.dataset]
        
        diffusion_model = initialize_diffusion_model(image_shape).to(device)
        diffusion_model.load_state_dict(torch.load(model_path))
        diffusion_model.eval()

        generate_samples(diffusion_model, args.dataset, batch_size=args.batch_size, num_steps=args.num_steps, output_path=args.output)

if __name__ == "__main__":
    main()
