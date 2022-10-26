import pickle
import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from tensorflow import math
import tensorflow as tf
import random
from matplotlib.colors import ListedColormap
from scipy import ndimage


# ------------------- Read data --------------------
def load_img(path, preprocess=True):
    filename = open(path, "rb")
    dataset = pickle.load(filename)
    filename.close()
    img = []
    for key in sorted(dataset.keys()):
        slice = dataset[key].pixel_array
        if preprocess:
            slice = preprocess_img(dataset[key], slice)
        img.append(slice)
    img = np.array(img)
    img = np.moveaxis(img, 0, -1)
    x_pixel = float(dataset[key].PixelSpacing[0])
    y_pixel = float(dataset[key].PixelSpacing[1])
    spacing = np.array([x_pixel, y_pixel, float(dataset[key].SliceThickness)])
    return img, spacing


def load_masks(path):
    data = np.load(path)
    mask = data['mask'].astype(int)
    mask = np.moveaxis(mask, 0, 1)
    return mask


# ----------------- Preprocessing ------------------
def transform_to_hu(dicom_image, np_image):
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope
    hu_image = np_image * slope + intercept
    return hu_image


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


def preprocess_img(dicom_slice, np_img):
    hu_image = transform_to_hu(dicom_slice, np_img)
    img = window_image(hu_image, -600, 1500)
    return img


def resample(img, spacing):
    image_shape = np.array([img.shape[0], img.shape[1], img.shape[2]])
    new_shape = np.round(image_shape * spacing)
    resize_factor = new_shape / image_shape
    resampled_image = ndimage.interpolation.zoom(img, resize_factor)
    return resampled_image


def padding(img, size):
    pad_x = (0, 0)
    pad_y = (0, 0)
    pad_z = (0, 0)
    if img.shape[0] < size:
        pad_x = (size - img.shape[0], 0)
    if img.shape[1] < size:
        pad_y = (size - img.shape[1], 0)
    if img.shape[2] < size:
        pad_z = (size - img.shape[2], 0)
    padded_img = np.pad(img, (pad_x, pad_y, pad_z), 'constant')
    return padded_img


def padding_infer(img, size):
    # if we don't consider separate padding functions for the training and test,
    # the random crop in the training phase may include only zeros.
    pad_x = (0, 0)
    pad_y = (0, 0)
    pad_z = (0, 0)
    if img.shape[0] % size != 0:
        pad = ((img.shape[0] // size) + 1) * size - img.shape[0]
        pad_x = (pad, 0)
    if img.shape[1] % size != 0:
        pad = ((img.shape[1] // size) + 1) * size - img.shape[1]
        pad_y = (pad, 0)
    if img.shape[2] % size != 0:
        pad = ((img.shape[2] // size) + 1) * size - img.shape[2]
        pad_z = (pad, 0)
    padded_img = np.pad(img, (pad_x, pad_y, pad_z), 'constant')
    return padded_img


def crop_img(img, roi):
    start_x, end_x = img.shape[0] // 2 - roi[0] // 2, img.shape[0] // 2 + roi[0] // 2
    start_y, end_y = img.shape[1] // 2 - roi[1] // 2, img.shape[1] // 2 + roi[1] // 2
    start_z, end_z = img.shape[2] // 2 - roi[2] // 2, img.shape[2] // 2 + roi[2] // 2
    img = img[start_x:end_x, start_y:end_y, start_z:end_z]
    return img


def create_random_patches(image, num, size, x=None, y=None, z=None, model3d=False):
    if model3d:
        image = np.squeeze(image, axis=-1)
    if x is None and y is None and z is None:
        x_end = image.shape[0] - size - 1
        if x_end < 0:
            x = [0] * num
        else:
            x = random.sample(range(1, x_end), num)
        y_end = image.shape[1] - size - 1
        if y_end < 0:
            y = [0] * num
        else:
            y = random.sample(range(1, y_end), num)
        z_end = image.shape[2] - size - 1
        if z_end < 0:
            z = [0] * num
        else:
            z = random.sample(range(1, z_end), num)
    cropped = []
    for a, b, c in zip(x, y, z):
        img = image[a:a + size,
              b:b + size,
              c:c + size]
        if model3d:
            img = np.expand_dims(img, axis=-1)
        cropped.append(img)
    return cropped, x, y, z


# ------------------ Plot images -------------------
def plot_3d(image, step_size=1):
    verts, faces, _, _ = marching_cubes(image, step_size=step_size, allow_degenerate=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.90)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()


def plot_overlay(image, mask, ground_truth, axis, slice):
    plt.figure()
    cmap1 = ListedColormap(['none', 'green'])
    cmap2 = ListedColormap(['none', 'red'])
    if axis == 0:
        img = image[slice, :, :]
        gt = ground_truth[slice, :, :]
        msk = mask[slice, :, :]
    elif axis == 1:
        img = image[:, slice, :]
        gt = ground_truth[:, slice, :]
        msk = mask[:, slice, :]
    else:
        img = image[:, :, slice]
        gt = ground_truth[:, :, slice]
        msk = mask[:, :, slice]
    plt.imshow(img, 'gray', interpolation='none')
    plt.imshow(gt, cmap2, interpolation='none', alpha=0.5)
    plt.imshow(msk, cmap1, interpolation='none', alpha=0.5)
    plt.show()


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(dim, volume, gt, mask=None):
    cmap1 = ListedColormap(['none', 'red'])
    cmap2 = ListedColormap(['none', 'blue'])
    remove_keymap_conflicts({'left', 'right'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.gt = gt
    ax.mask = mask
    if dim == 0:
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index, :, :], cmap='gray')
        ax.imshow(gt[ax.index, :, :], cmap1, interpolation='none', alpha=0.5)
        if mask is not None:
            ax.imshow(mask[ax.index, :, :], cmap2, interpolation='none', alpha=0.5)
    elif dim == 1:
        ax.index = volume.shape[1] // 2
        ax.imshow(volume[:, ax.index, :], cmap='gray')
        ax.imshow(gt[:, ax.index, :], cmap1, interpolation='none', alpha=0.5)
        if mask is not None:
            ax.imshow(mask[:, ax.index, :], cmap2, interpolation='none', alpha=0.5)
    else:
        ax.index = volume.shape[2] // 2
        ax.imshow(volume[:, :, ax.index], cmap='gray')
        ax.imshow(gt[:, :, ax.index], cmap1, interpolation='none', alpha=0.5)
        if mask is not None:
            ax.imshow(mask[:, :, ax.index], cmap2, interpolation='none', alpha=0.5)
    if dim == 0:
        fig.canvas.mpl_connect('key_press_event', process_key_0)
    elif dim == 1:
        fig.canvas.mpl_connect('key_press_event', process_key_1)
    else:
        fig.canvas.mpl_connect('key_press_event', process_key_2)


def process_key_0(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    dim = 0
    if event.key == 'left':
        previous_slice(ax, dim)
    elif event.key == 'right':
        next_slice(ax, dim)
    fig.canvas.draw()


def process_key_1(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    dim = 1
    if event.key == 'left':
        previous_slice(ax, dim)
    elif event.key == 'right':
        next_slice(ax, dim)
    fig.canvas.draw()


def process_key_2(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    dim = 2
    if event.key == 'left':
        previous_slice(ax, dim)
    elif event.key == 'right':
        next_slice(ax, dim)
    fig.canvas.draw()


def previous_slice(ax, dim):
    volume = ax.volume
    gt = ax.gt
    mask = ax.mask
    if dim == 0:
        ax.index = (ax.index - 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index, :, :])
        ax.images[1].set_array(gt[ax.index, :, :])
        if mask is not None:
            ax.images[2].set_array(mask[ax.index, :, :])
    elif dim == 1:
        ax.index = (ax.index - 1) % volume.shape[1]
        ax.images[0].set_array(volume[:, ax.index, :])
        ax.images[1].set_array(gt[:, ax.index, :])
        if mask is not None:
            ax.images[2].set_array(mask[:, ax.index, :])
    elif dim == 2:
        ax.index = (ax.index - 1) % volume.shape[2]
        ax.images[0].set_array(volume[:, :, ax.index])
        ax.images[1].set_array(gt[:, :, ax.index])
        if mask is not None:
            ax.images[2].set_array(mask[:, :, ax.index])


def next_slice(ax, dim):
    volume = ax.volume
    gt = ax.gt
    mask = ax.mask
    if dim == 0:
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index, :, :])
        ax.images[1].set_array(gt[ax.index, :, :])
        if mask is not None:
            ax.images[2].set_array(mask[ax.index, :, :])
    elif dim == 1:
        ax.index = (ax.index + 1) % volume.shape[1]
        ax.images[0].set_array(volume[:, ax.index, :])
        ax.images[1].set_array(gt[:, ax.index, :])
        if mask is not None:
            ax.images[2].set_array(mask[:, ax.index, :])
    elif dim == 2:
        ax.index = (ax.index + 1) % volume.shape[2]
        ax.images[0].set_array(volume[:, :, ax.index])
        ax.images[1].set_array(gt[:, :, ax.index])
        if mask is not None:
            ax.images[2].set_array(mask[:, :, ax.index])


# -------------- Evaluation metrics ----------------
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = math.reduce_sum(math.abs(y_true * y_pred), axis=[-3, -2, -1])
    first_sum = math.reduce_sum(math.square(y_true), axis=[-3, -2, -1])
    second_sum = math.reduce_sum(math.square(y_pred), axis=[-3, -2, -1])
    dn = math.add(first_sum, second_sum)
    epsilon = 1e-8
    f_dn = math.add(dn, epsilon)
    dice = math.reduce_mean(2 * intersection / f_dn)
    return dice


def loss_gt(e=1e-8):
    def loss_gt_(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = math.reduce_sum(math.abs(y_true * y_pred), axis=[-3, -2, -1])
        first_sum = math.reduce_sum(math.square(y_true), axis=[-3, -2, -1])
        second_sum = math.reduce_sum(math.square(y_pred), axis=[-3, -2, -1])
        dn = math.add(first_sum, second_sum)
        epsilon = 1e-8
        f_dn = math.add(dn, epsilon)
        dice = math.reduce_mean(2 * intersection / f_dn)
        return 1 - dice

    return loss_gt_


# ------------------ Checkpoints -------------------
def save_best_model(path, monitor='val_dice_coefficient', mode='max'):
    checkpoit_best = ModelCheckpoint(path,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     monitor=monitor,
                                     mode=mode,
                                     verbose=1)
    return checkpoit_best


def scheduler(epoch, lr):
    if epoch % 5 == 4:
        new_lr = lr * 0.7
    else:
        new_lr = lr
    return new_lr


# ------------------- Generator --------------------
class CustomGenerator(keras.utils.Sequence):

    def __init__(self, data_paths, label_paths, batch_size, size, num_patch, model3d, preprocess, resampling):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.size = size
        self.num_patch = num_patch
        self.model3d = model3d
        self.preprocess = preprocess
        self.resampling = resampling

    def __len__(self):
        return (np.ceil(len(self.data_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.data_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.label_paths[idx * self.batch_size: (idx + 1) * self.batch_size]

        imgs = []
        labels = []
        for img_path, label_path in zip(batch_x, batch_y):
            img, spacing = load_img(img_path, self.preprocess)
            mask = load_masks(label_path)
            if self.resampling:
                # resample images and masks to have a standard size of 1*1*1
                img = resample(img, spacing)
                mask = resample(mask, spacing)
            # add padding as the shape of some images is smaller than the desired roi
            img = padding(img, self.size)
            mask = padding(mask, self.size)
            if self.model3d:
                img = np.expand_dims(img, axis=-1)
                mask = np.expand_dims(mask, axis=-1)
            cropped_img, x, y, z = create_random_patches(img, self.num_patch, self.size, None, None, None, self.model3d)
            cropped_mask, x, y, z = create_random_patches(mask, self.num_patch, self.size, x, y, z, self.model3d)
            imgs.extend(cropped_img)
            labels.extend(cropped_mask)

        return np.array(imgs), np.array(labels)


# ------------------- Inference --------------------
def inference(model, img, roi):
    pad_img = padding_infer(img, roi[0])
    mask = np.zeros(pad_img.shape)
    shape = pad_img.shape
    for a in range(shape[0] // roi[0]):
        start_a = a * roi[0]
        end_a = (a + 1) * roi[0]
        for b in range(shape[1] // roi[1]):
            start_b = b * roi[1]
            end_b = (b + 1) * roi[1]
            for c in range(shape[2] // roi[2]):
                start_c = c * roi[2]
                end_c = (c + 1) * roi[2]
                cropped_img = pad_img[start_a:end_a, start_b:end_b, start_c:end_c]
                cropped_img = cropped_img[np.newaxis, ...]
                prediction = model.predict(cropped_img)
                prediction = np.squeeze(prediction)
                prediction = np.where(prediction >= 0.5, 1, 0)
                mask[start_a:end_a, start_b:end_b, start_c:end_c] = prediction

    mask = remove_padding(img, mask, roi[0]).astype(int)
    return mask


# ----------------- Post process -------------------
def remove_padding(img, mask, size):
    pad_x = 0
    pad_y = 0
    pad_z = 0
    if img.shape[0] % size != 0:
        pad_x = ((img.shape[0] // size) + 1) * size - img.shape[0]
    if img.shape[1] % size != 0:
        pad_y = ((img.shape[1] // size) + 1) * size - img.shape[1]
    if img.shape[2] % size != 0:
        pad_z = ((img.shape[2] // size) + 1) * size - img.shape[2]

    mask = mask[pad_x:, pad_y:, pad_z:]
    return mask
