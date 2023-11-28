import os
import sys
import glob
import cv2
import numpy as np
from argparse import Namespace
from numpy import ndarray
from imageio.core.util import Array
from imageio import imread
from cv2.mat_wrapper import Mat

from accuracy import AccuracyTracker
from solution.utils.utils import SIZE, get_eval_args


def load_ground_truth_image(image_path: str) -> ndarray:
    image: Array = imread(uri=image_path).astype(np.uint8)

    if len(image.shape) == 3:
        image = image[:, :, 0]

    resized_image: Mat = cv2.resize(
        image, tuple(SIZE),
        interpolation=cv2.INTER_AREA
    )
    resized_image: Mat = cv2.resize(
        resized_image, tuple(SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    output_image: ndarray = resized_image[np.newaxis, :, :]

    return output_image


def get_score(image, groundTruth):
    acc_tracker: AccuracyTracker = AccuracyTracker(n_classes=14)
    gt_array: ndarray = load_ground_truth_image(imagePath=groundTruth)
    out_array: ndarray = load_ground_truth_image(imagePath=image)
    acc_tracker.update(gt_array, out_array)
    acc_tracker.get_scores()
    return acc_tracker.mean_dice


def main(args: Namespace):
    segmentation_map_directory = args.image_dir
    ground_truth_directory = args.gt

    segmentation_maps = \
        sorted(glob.glob(os.path.join(segmentation_map_directory, '*.png')))

    ground_truths = \
        sorted(glob.glob(os.path.join(ground_truth_directory, '*.png')))

    if len(segmentation_maps) != len(ground_truths):
        print("The number of seg maps and gt images does not match.")
        sys.exit(1)

    total_score = 0
    image_count = len(segmentation_maps)

    for image, ground_truth in zip(segmentation_maps, ground_truths):
        score = get_score(image, ground_truth)
        total_score += score

    average_score = total_score / image_count
    print(f"{average_score}")


if __name__ == '__main__':
    args: Namespace = get_eval_args()
    main(args)
