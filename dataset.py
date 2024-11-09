import medmnist
import torch
from torchvision import transforms

def initialize_datasets_and_dataloaders(dataset_name, batch_size):

    if dataset_name == 'PathMNIST':
        medmnist_dataset = medmnist.PathMNIST
        image_shape = (3, 28, 28)
    elif dataset_name == 'ChestMNIST':
        medmnist_dataset = medmnist.ChestMNIST
        image_shape = (1, 28, 28)
    elif dataset_name == 'BloodMNIST':
        medmnist_dataset = medmnist.BloodMNIST
        image_shape = (3, 28, 28)
    elif dataset_name == 'DermaMNIST':
        medmnist_dataset = medmnist.DermaMNIST
        image_shape = (3, 28, 28)

    transform = transforms.Compose([
                    transforms.ToTensor()
                ])

    medmnist_train_dataset = medmnist_dataset(
                                split='train',
                                transform=transform,
                                download=True)

    medmnist_val_dataset = medmnist_dataset(
                                split='val',
                                transform=transform,
                                download=True)
    
    medmnist_test_dataset = medmnist_dataset(
                                split='test',
                                transform=transform,
                                download=True)

    medmnist_train_dataloader = torch.utils.data.DataLoader(
                                                    medmnist_train_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    num_workers=4)

    medmnist_val_dataloader = torch.utils.data.DataLoader(
                                                    medmnist_val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    
    medmnist_test_dataloader = torch.utils.data.DataLoader(
                                                    medmnist_test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    
    return medmnist_train_dataset, medmnist_val_dataset, medmnist_test_dataset, medmnist_train_dataloader, medmnist_val_dataloader, medmnist_test_dataloader, image_shape