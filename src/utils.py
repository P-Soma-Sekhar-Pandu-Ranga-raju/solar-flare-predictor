from PIL import Image


# -------------------------------------------------
# 1. Check if an image is flagged / noisy (EXIF)
# -------------------------------------------------
def is_flagged_image(img_path):
    """
    Checks whether an image is marked as 'flagged'
    in its EXIF metadata.

    Used during training to handle noisy SDO images.
    """
    try:
        img = Image.open(img_path)
        exif = img.getexif()

        for _, value in exif.items():
            if isinstance(value, str) and "flagged" in value.lower():
                return True
    except Exception:
        # If EXIF is missing or unreadable, treat as not flagged
        pass

    return False


# -------------------------------------------------
# 2. Convert peak_flux → Binary + Flare class
# -------------------------------------------------
def interpret_prediction(peak_flux):
    """
    Converts predicted peak X-ray flux (W/m^2) into:
    - Binary flare warning
    - Flare class (B / C / M / X)

    Standard GOES thresholds are used.
    """

    # Flare class thresholds
    if peak_flux < 1e-7:
        flare_class = "B"
    elif peak_flux < 1e-6:
        flare_class = "C"
    elif peak_flux < 1e-5:
        flare_class = "M"
    else:
        flare_class = "X"

    # Binary classification
    if flare_class in ["M", "X"]:
        binary = "✅ Major Flare"
    else:
        binary = "❌ No Major Flare"

    return binary, flare_class
