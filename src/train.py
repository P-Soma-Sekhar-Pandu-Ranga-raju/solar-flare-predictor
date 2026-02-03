import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SDODataset
from model import SDOModel


def main():
    # -------------------------
    # Device
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    train_dataset = SDODataset("data/training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,      # IMPORTANT for Windows (prevents silent hangs)
        pin_memory=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = SDOModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()

    # -------------------------
    # Training loop
    # -------------------------
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\n🚀 Starting Epoch {epoch+1}/{epochs}")

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            unit="batch"
        )

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm bar
            progress_bar.set_postfix(
                loss=loss.item()
            )

            # Extra heartbeat (optional but useful)
            if batch_idx == 0:
                print("  ✔ First batch processed")

        avg_loss = running_loss / len(train_loader)
        print(
            f"✅ Epoch [{epoch+1}/{epochs}] "
            f"Train MAE (log): {avg_loss:.4f}"
        )

    # -------------------------
    # Save model
    # -------------------------
    torch.save(model.state_dict(), "models/sdo_model.pth")
    print("💾 Model saved to models/sdo_model.pth")


if __name__ == "__main__":
    main()
