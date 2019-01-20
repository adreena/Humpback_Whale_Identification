
from matplotlib import pyplot as plt
import numpy as np
import scipy.misc
import csv
import os


# Get next batch of data
#################################################################################
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

# Download VGG
#################################################################################
def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))
