import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure standard output handles UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import config
from dataset import crop_image_from_gray, preprocess_ben_graham
from model import get_model

# Clinical guidance mapping for each DR severity level
CLINICAL_RECOMMENDATIONS = {
    0: {
        "status": "Healthy / No DR",
        "severity": "Normal",
        "action": "Routine annual dilated eye exam recommended."
    },
    1: {
        "status": "Mild Non-Proliferative DR",
        "severity": "Low Risk",
        "action": "Microaneurysms detected. Follow-up eye exam in 6-12 months; monitor blood glucose levels."
    },
    2: {
        "status": "Moderate Non-Proliferative DR",
        "severity": "Medium Risk",
        "action": "Multiple microaneurysms, blot hemorrhages, or hard exudates. Specialist consultation recommended in 3-6 months."
    },
    3: {
        "status": "Severe Non-Proliferative DR",
        "severity": "High Risk",
        "action": "Extensive hemorrhages or venous beading. Urgent ophthalmology referral required within 2-4 weeks to prevent vision loss."
    },
    4: {
        "status": "Proliferative Diabetic Retinopathy",
        "severity": "Critical Risk",
        "action": "Neovascularization or vitreous hemorrhage detected. Immediate retinal specialist intervention / laser photocoagulation or anti-VEGF therapy required."
    }
}

def load_inference_model(weights_path=config.MODEL_SAVE_PATH, device=config.DEVICE):
    """
    Loads the trained NetraVision model weights into eval mode.
    """
    model = get_model(num_classes=config.NUM_CLASSES)

    if not os.path.exists(weights_path):
        print(f"[WARNING] Model weights not found at: {weights_path}")
        print("          Inference will run with initial weights (for pipeline testing).")
        print("          Train the model first using 'python train.py' to generate trained weights.\n")
    else:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_qwk = checkpoint.get('val_qwk', 'N/A')
            print(f"[INFO] Loaded trained weights from: {weights_path} (Best Val QWK: {val_qwk})")
        else:
            model.load_state_dict(checkpoint)
            print(f"[INFO] Loaded weights from: {weights_path}")

    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=config.IMG_SIZE):
    """
    Reads and applies Ben Graham's fundus preprocessing to match training data.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_ben_graham(img_rgb, img_size=img_size)

    # Tensor conversion & ImageNet normalization
    tensor = torch.from_numpy(processed_img.transpose((2, 0, 1))).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    # Add batch dimension (1, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor, img_rgb, processed_img

def predict_single_image(image_path, weights_path=config.MODEL_SAVE_PATH):
    device = config.DEVICE
    model = load_inference_model(weights_path, device)
    tensor, original_img, processed_img = preprocess_image(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

    rec = CLINICAL_RECOMMENDATIONS[predicted_class]

    print("=" * 65)
    print(f"👁️  NETRAVISION DIABETIC RETINOPATHY ANALYSIS")
    print("=" * 65)
    print(f"File:             {os.path.basename(image_path)}")
    print(f"Predicted Grade:  Class {predicted_class} - {rec['status']}")
    print(f"Severity Level:   {rec['severity']}")
    print(f"Confidence:       {confidence * 100:.2f}%")
    print("-" * 65)
    print("Class Probabilities Breakdown:")
    for cls_idx in range(config.NUM_CLASSES):
        pct = probs[cls_idx] * 100
        bar = "█" * int(pct / 5)
        marker = " <== PREDICTED" if cls_idx == predicted_class else ""
        print(f"  Class {cls_idx} ({config.CLASS_NAMES[cls_idx]:<18}): {pct:5.1f}%  {bar:<20}{marker}")
    print("-" * 65)
    print(f"Clinical Guidance:\n  {rec['action']}")
    print("=" * 65)

    return {
        "predicted_class": predicted_class,
        "class_name": config.CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "recommendation": rec
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Diabetic Retinopathy grade for a retinal fundus image.")
    parser.add_argument("--image", type=str, default=None, help="Path to input retinal fundus image (.png, .jpg)")
    parser.add_argument("--weights", type=str, default=config.MODEL_SAVE_PATH, help="Path to model checkpoint .pth")
    parser.add_argument("--sample", action="store_true", help="Pick a random image from test_images/ for demonstration")
    args = parser.parse_args()

    target_image = args.image

    if args.sample or target_image is None:
        # If no image specified, pick a sample from test_images or train_images
        sample_dir = config.TEST_IMAGES_DIR if os.path.exists(config.TEST_IMAGES_DIR) else config.TRAIN_IMAGES_DIR
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            import random
            chosen = random.choice(sample_files)
            target_image = os.path.join(sample_dir, chosen)
            print(f"[INFO] No image provided. Selected sample: {target_image}\n")
        else:
            print("[ERROR] No images found to test.")
            sys.exit(1)

    predict_single_image(target_image, weights_path=args.weights)
