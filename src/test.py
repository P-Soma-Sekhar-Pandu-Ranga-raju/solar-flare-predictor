import torch
from torch.utils.data import DataLoader
from dataset import SDODataset
from model import SDOModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    test_dataset = SDODataset("data/test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    model = SDOModel().to(device)
    model.load_state_dict(
        torch.load("models/sdo_model.pth", map_location=device)
    )
    model.eval()

    errors = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            pred_log = model(x)
            pred = 10 ** pred_log
            true = 10 ** y

            errors.append(torch.abs(pred - true))

    mae = torch.cat(errors).mean().item()
    print("✅ Test MAE (physical units):", mae)

if __name__ == "__main__":
    main()
