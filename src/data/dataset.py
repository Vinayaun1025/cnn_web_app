from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloader(path, transform, batch_size, shuffle=True):
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
