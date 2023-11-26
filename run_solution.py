import os
import pkg_resources
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from argparse import Namespace
from PIL import Image
from torchvision.transforms import transforms
from imageio import imread
from cv2.mat_wrapper import Mat

from utils import DEVICE, get_solution_args
from fanet import FANet


SIZE: List[int] = [512, 512]


def load_model():
    # TODO make sure this is the path to our custom model pkl
    modelPath: str = "model.pkl"
    # TODO should the entire eval process be in this context manager?
    with pkg_resources.resource_stream(__name__, modelPath) as model_file:
        model: FANet = FANet()
        device = torch.device("cuda")
        model.to(device)
        model.load_state_dict(
                state_dict=torch.load(
                    f=model_file,
                    map_location=DEVICE
                ),
                strict=False
        )
        return model


def loadImageToTensor(imagePath: str) -> torch.Tensor:
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STANDARD_DEVIATION: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    image = imread(uri=imagePath)
    resizedImage: Mat = \
        cv2.resize(image, tuple(SIZE), interpolation=cv2.INTER_AREA)

    imageTensor = transforms.ToTensor()(resizedImage)
    imageTensor = transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)(
        imageTensor
    )

    imageTensor = imageTensor.unsqueeze(0)

    return imageTensor


def main(args: Namespace) -> None:
    image_files: List[str] = os.listdir(args.input)

    model = load_model()
    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm Up
    for image_file in image_files[:15]:
        input_image_path: str = os.path.join(args.input, image_file)
        imageTensor: torch.Tensor = \
            loadImageToTensor(imagePath=input_image_path)
        imageTensor = imageTensor.to(DEVICE)
        outTensor: torch.Tensor = model(imageTensor)

    time = 0
    with torch.no_grad():
        for image_file in image_files:
            input_image_path: str = os.path.join(args.input, image_file)
            output_image_path: str = os.path.join(args.output, image_file)
            imageTensor: torch.Tensor = \
                loadImageToTensor(imagePath=input_image_path)
            imageTensor = imageTensor.to(DEVICE)
            start.record()
            outTensor: torch.Tensor = model(imageTensor)
            end.record()
            torch.cuda.synchronize()

            time += start.elapsed_time(end)

            outTensor: torch.Tensor = F.interpolate(
                outTensor, SIZE, mode="bilinear", align_corners=True
            )

            outArray: np.ndarray = outTensor.cpu().data.max(1)[1].numpy()
            outArray: np.ndarray = outArray.astype(np.uint8)

            outImage: np.ndarray = np.squeeze(outArray, axis=0)
            outImage = Image.fromarray(outImage, mode='L')
            outImage.save(output_image_path)

    print(time/1000)

    del model
    del imageTensor
    del outTensor

    torch.cuda.empty_cache()


if __name__ == '__main__':
    args: Namespace = get_solution_args()
    main(args)
