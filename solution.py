import os
import pkg_resources
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from argparse import Namespace
from PIL import Image
from torchvision.transforms import transforms
from imageio import imread
from cv2.mat_wrapper import Mat

from utils.utils import (
    DEVICE, SIZE, MEAN, STANDARD_DEVIATION, MODEL_FILE, get_solution_args
)
from models.fanet import FANet


def load_model():
    # TODO should the entire eval process be in this context manager?
    with pkg_resources.resource_stream(__name__, MODEL_FILE) as model_file:
        model: FANet = FANet()
        model.to(DEVICE)
        model.load_state_dict(
                state_dict=torch.load(
                    f=model_file,
                    map_location=DEVICE
                ),
                strict=False
        )
        return model


def load_image_to_tensor(image_path: str) -> torch.Tensor:
    image = imread(uri=image_path)
    resized_image: Mat = \
        cv2.resize(image, tuple(SIZE), interpolation=cv2.INTER_AREA)

    image_tensor = transforms.ToTensor()(resized_image)
    image_tensor = transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)(
        image_tensor
    )

    return image_tensor.unsqueeze(0)


def main(args: Namespace) -> None:
    image_files: list[str] = os.listdir(args.input)

    model = load_model()
    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm Up
    for image_file in image_files[:15]:
        input_image_path: str = os.path.join(args.input, image_file)
        image_tensor: torch.Tensor = \
            load_image_to_tensor(image_path=input_image_path)
        image_tensor = image_tensor.to(DEVICE)
        out_tensor: torch.Tensor = model(image_tensor)

    time = 0
    with torch.no_grad():
        for image_file in image_files:
            input_image_path: str = os.path.join(args.input, image_file)
            output_image_path: str = os.path.join(args.output, image_file)
            image_tensor: torch.Tensor = \
                load_image_to_tensor(image_path=input_image_path)
            image_tensor = image_tensor.to(DEVICE)
            start.record()
            out_tensor: torch.Tensor = model(image_tensor)
            end.record()
            torch.cuda.synchronize()

            time += start.elapsed_time(end)

            out_tensor: torch.Tensor = F.interpolate(
                out_tensor, SIZE, mode="bilinear", align_corners=True
            )

            out_array: np.ndarray = out_tensor.cpu().data.max(1)[1].numpy()
            out_array: np.ndarray = out_array.astype(np.uint8)

            out_image: np.ndarray = np.squeeze(out_array, axis=0)
            out_image = Image.fromarray(out_image, mode='L')
            out_image.save(output_image_path)

    print(time/1000)

    del model
    del image_tensor
    del out_tensor

    torch.cuda.empty_cache()


if __name__ == '__main__':
    args: Namespace = get_solution_args()
    main(args)
