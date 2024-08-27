import argparse
import os
import pandas as pd
import torch
import cv2
from pytorch_lightning import seed_everything
from src.config import Config
from src.augmentations import get_transforms
from src.lightning_module import PlanetModule


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Inference script for the Planet dataset')
    parser.add_argument('--model-folder', type=str, default='model_weights', help='Path to model weights')
    parser.add_argument('--input-folder', type=str, default='inference_images', help='Path to input images')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to CSV file predictions')
    parser.add_argument('--config-file', type=str, default='config/config.yaml', help='Path to the config file')
    return parser.parse_args()


def load_model(config: Config, model_path: str) -> PlanetModule:
    """
    Load the trained model from a checkpoint.

    Args:
        config (Config): Configuration object.
        model_path (str): Path to the model checkpoint.

    Returns:
        PlanetModule: The loaded model.
    """
    model = PlanetModule.load_from_checkpoint(model_path, config=config)
    model.eval()
    return model


def prepare_image(image_path: str, config: Config):
    """
    Prepare the image for inference.

    Args:
        image_path (str): Path to the input image file.
        config (Config): Configuration object.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms(
        aug_config=config.augmentation_params,
        width=config.data_config.width,
        height=config.data_config.height,
        flags=None,
    )
    data_list = transform(image=image)
    image = data_list['image']
    return torch.tensor(image).unsqueeze(0)


@torch.inference_mode()
def run_inference(model: PlanetModule, image: torch.Tensor, config: Config, image_path: str):
    """
    Run inference on a single image and returns the predictions.

    Args:
        model (PlanetModule): The trained model.
        image (torch.Tensor): Preprocessed image tensor.
        config (Config): Configuration object.
        image_path (str): Path to the input image file.

    Returns:
        dict: Predictions with image name and tags.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)

    logits = model(image)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    label_names = list(config.label_encoder.keys())
    threshold = 0.5

    # Get the labels above the threshold
    tags = [label for label, prob in zip(label_names, probs) if prob > threshold]

    # Extract the image name
    image_name = os.path.basename(image_path)

    return {'image_name': image_name, 'tags': ' '.join(tags)}


def main():
    args = parse_args()
    config = Config.from_yaml(args.config_file)
    seed_everything(config.seed, workers=True)

    # Find the model weights file in the specified folder
    model_weights_files = os.listdir(args.model_folder)
    if not model_weights_files:
        raise FileNotFoundError(f'No model weights found in the directory: {args.model_folder}')
    model_path = os.path.join(args.model_folder, model_weights_files[0])

    model = load_model(config, model_path)
    predictions = []
    for image_name in os.listdir(args.input_folder):
        image_path = os.path.join(args.input_folder, image_name)
        image = prepare_image(image_path, config)
        prediction = run_inference(model, image, config, image_path)
        predictions.append(prediction)

    df = pd.DataFrame(predictions)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
