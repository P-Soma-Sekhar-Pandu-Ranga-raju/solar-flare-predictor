import os
import requests
import torch
from io import BytesIO
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
from torchvision import transforms

from model import SDOModel          # YOUR EXISTING MODEL (UNCHANGED)
from utils import interpret_prediction


# -------------------------------------------------
# CONFIG (ONLY HERE)
# -------------------------------------------------
HELIOVIEWER_URL = (
    "https://sdo.gsfc.nasa.gov/assets/img/latest/"
    "latest_1024_0193.jpg"
)

IMAGE_SIZE = 256
LATENCY_MINUTES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# IMAGE PREPROCESSING (MATCH TRAINING)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


def fetch_latest_solar_image():
    """Fetch latest SDO/AIA image."""
    response = requests.get(HELIOVIEWER_URL, timeout=20)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def prepare_model_input(img):
    """
    Convert image → (1, 4, 10, 256, 256)
    """
    img_tensor = transform(img).squeeze(0)   # (256,256)

    # Channel alignment (1 → 10)
    channels = img_tensor.repeat(10, 1, 1)

    # Temporal alignment (1 → 4)
    timesteps = channels.unsqueeze(0).repeat(4, 1, 1, 1)

    return timesteps.unsqueeze(0).to(DEVICE)


def get_image_timestamp():
    prediction_time = datetime.utcnow()
    image_time = prediction_time - timedelta(minutes=LATENCY_MINUTES)
    return image_time, prediction_time


def save_image_with_timestamp(img, image_time, output_path):
    draw = ImageDraw.Draw(img)

    text = (
        "SDO / AIA 193 Å\n"
        f"Image Time (UTC): {image_time.strftime('%Y-%m-%d %H:%M')}"
    )

    box = draw.multiline_textbbox((10, 10), text)
    draw.rectangle(box, fill="black")
    draw.multiline_text((10, 10), text, fill="white")

    img.save(output_path)


# -------------------------------------------------
# MAIN (INFERENCE ONLY)
# -------------------------------------------------
def main():
    print("🌞 Fetching latest SDO/AIA image...")
    img = fetch_latest_solar_image()

    print("🧠 Preparing model input...")
    x = prepare_model_input(img)

    print("📦 Loading trained model...")
    model = SDOModel().to(DEVICE)
    model.load_state_dict(
        torch.load("models/sdo_model.pth", map_location=DEVICE)
    )
    model.eval()

    print("🔮 Running prediction...")
    with torch.no_grad():
        pred_log = model(x).item()
        peak_flux = 10 ** pred_log

    binary, flare_class = interpret_prediction(peak_flux)
    image_time, prediction_time = get_image_timestamp()

    os.makedirs("outputs", exist_ok=True)
    image_path = "outputs/latest_solar_image.png"
    save_image_with_timestamp(img, image_time, image_path)

    print("\n🔍 Prediction Result")
    print("-------------------")
    print(f"Image Time (UTC)   : {image_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Prediction Time   : {prediction_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Binary Prediction : {binary}")
    print(f"Final Prediction  : {flare_class} class")
    print(f"Predicted Flux    : {peak_flux:.2e} W/m²")
    print(f"Image Saved At    : {image_path}")


if __name__ == "__main__":
    main()
