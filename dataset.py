import medmnist
from torchvision import transforms
import torch

medmnist_mapping = {
    'PathMNIST': (medmnist.PathMNIST, (3, 28, 28), 1, 'weights/pathmnist_model.pth'),
    'ChestMNIST': (medmnist.ChestMNIST, (1, 28, 28), 14, 'weights/chestmnist_model.pth'),
    'BloodMNIST': (medmnist.BloodMNIST, (3, 28, 28), 1, 'weights/bloodmnist_model.pth'),
    'DermaMNIST': (medmnist.DermaMNIST, (3, 28, 28), 1, 'weights/dermamnist_model.pth')
}

def initialize_datasets(dataset_name):
    medmnist_dataset, image_shape, y_dim, model_path = medmnist_mapping.get(dataset_name, (None, None, None))
    if medmnist_dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = medmnist_dataset(split='train', transform=transform, download=True)
    val_dataset = medmnist_dataset(split='val', transform=transform, download=True)
    test_dataset = medmnist_dataset(split='test', transform=transform, download=True)

    return train_dataset, val_dataset, test_dataset, image_shape, y_dim, model_path

def create_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
