import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from ENNOS.dataset import OrganoidDataset
from ENNOS.networks.ennos_model import ENNOSModel
from ENNOS.utils import *

MODEL_DIR = "ENNOS/weights/"
IMAGES_PATH = "ENNOS/dataset/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
EPOCHS = 50
TRANSFORM = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

def main():
    test_dataset = OrganoidDataset(images_path=IMAGES_PATH, mask_path=IMAGES_PATH, mode='test', transform=TRANSFORM)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ENNOSModel(input_shape=1, hidden_units=64, output_shape=1, device=DEVICE, model_name=os.path.join(MODEL_DIR, 'univr_pronet_model.pth')).to(DEVICE)
    load_model(model=model, model_name=os.path.join(MODEL_DIR, 'univr_ennos_50_model.pth'), device=DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    start_time = timer()
    testing_loop(model=model, dataloader=test_loader, loss_fn=criterion, device=DEVICE, test=True, show=True)
    end_time = timer()
    print('\n')
    print_execution_time(start=start_time, end=end_time, device=DEVICE)
    print('Testing completed.')

if __name__ == "__main__":
    main()