
from matplotlib import pyplot as plt
import numpy as np
import scipy.misc
import csv
import os


def get_next_batch(image_path, label_file, batch_size, image_shape):
    """
    Creates batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return: batch of images
    """

    with open(label_file, "rb") as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]

    # TODO need to shuffle
    for batch_i in range(0, len(data), batch_size):
        images = []
        labels = []
        for image_file in data[batch_i:batch_i + batch_size]:
            label = image_file['Id']
            image = scipy. misc.imresize(scipy.misc.imread(os.path.join(image_path, image_file['Image'])), image_shape)

            images.append(image)
            labels.append(label)

        yield np.array(images), np.array(labels)
