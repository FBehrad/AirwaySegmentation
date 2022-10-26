import glob
import matplotlib
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.optimizers import Adam, SGD
from utils import load_img, load_masks, save_best_model, scheduler, CustomGenerator, plot_3d, loss_gt, dice_coefficient
from utils import plot_overlay, inference, multi_slice_viewer, resample
from model import unet, unet3d
import os
import datetime
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To disable tensorflow logs

    # ---------------------- Configs ----------------------
    train = True
    test = False
    roi = (256, 256, 256)
    roi_3d = (64, 64, 64, 1)
    learning_rate = 0.1
    num_patch = 2
    batch_size = 1
    model3d = False
    preprocess = True
    resampling = False  # apply resampling on images
    epochs = 2
    scheduler_type = 'linear'  # 'CosineDecay' or 'linear'
    decay_steps = 7  # In Cosine decay learning rate scheduler: number of steps to decay over
    alpha = 0.01  # minimum learning rate value= alpha * learning rate
    pretrained = False
    save_path = 'models/best2'
    load_path = 'models/best2'
    train_size = 0.8

    # ----------------- Create data paths -----------------
    address = '.\\Behrad'
    image_pattern = address + '/*/DICOM/*.pickle'
    mask_pattern = address + '/*/Airway_mask/*.npz'
    img_paths = glob.glob(image_pattern)
    mask_paths = glob.glob(mask_pattern)

    if train:
        # ----------------- Define checkpoints ------------------
        checkpoint_best = save_best_model(save_path)
        if scheduler_type == 'linear':
            reduce_lr = LearningRateScheduler(scheduler)
        else:
            reduce_lr = LearningRateScheduler(
                CosineDecayRestarts(initial_learning_rate=learning_rate, alpha=alpha, first_decay_steps=decay_steps))
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # -------------------- Define models --------------------
        if model3d:
            model = unet3d(input_size=roi_3d)
        else:
            model = unet(input_size=roi)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=[loss_gt()], metrics=[dice_coefficient])

        if pretrained:
            model.load_weights(load_path)

        # model.summary()

        # -------------------- Start training --------------------

        X_train, X_rem, y_train, y_rem = train_test_split(img_paths, mask_paths, train_size=train_size)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

        generator = CustomGenerator(X_train, y_train, batch_size=batch_size, size=roi[0], num_patch=num_patch,
                                    model3d=model3d, preprocess=preprocess, resampling=resampling)

        validation_generator = CustomGenerator(X_valid, y_valid, batch_size=batch_size, size=roi[0],
                                               num_patch=num_patch, model3d=model3d, preprocess=preprocess,
                                               resampling=resampling)

        test_generator = CustomGenerator(X_test, y_test, batch_size=batch_size, size=roi[0], num_patch=num_patch,
                                         model3d=model3d, preprocess=preprocess, resampling=resampling)

        history = model.fit(generator, validation_data=validation_generator, steps_per_epoch=int(len(X_train)),
                            epochs=epochs, verbose=1, callbacks=[checkpoint_best, reduce_lr,
                                                                 tensorboard_callback])

        score = model.evaluate(test_generator, batch_size=batch_size, callbacks=tensorboard_callback)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout(pad=3.0)
        ax1.plot(history.history['dice_coefficient'])
        ax1.plot(history.history['val_dice_coefficient'])
        ax1.set_title('Model dice')
        ax1.set(xlabel='epoch', ylabel='dice')
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model loss')
        ax2.set(xlabel='epoch', ylabel='loss')
        ax1.legend(['train', 'val'], loc='upper left')
        ax2.legend(['train', 'val'], loc='upper left')
        text = f'Test set: Dice: {score[1]:.3f} , loss: {score[0]:.3f}'
        plt.figtext(0.5, 0, text, wrap=True, horizontalalignment='center', fontsize=10)
        plt.savefig('performance.png')
        plt.show()

    elif test:
        # -------------------- Load model and predict mask -------------------
        model = unet(input_size=roi)
        model.load_weights(load_path).expect_partial()
        # -------------------- Read image and add padding --------------------

        img, spacing = load_img(img_paths[7], preprocess)
        ground_truth = load_masks(mask_paths[7])

        if resampling:
            # resample images and masks to have a standard size of 1*1*1
            img = resample(img, spacing)
            mask = resample(ground_truth, spacing)

        mask = inference(model, img, roi)

        # interactive plot
        matplotlib.use('TkAgg')
        multi_slice_viewer(2, img, ground_truth, mask)
        plt.show(block=True)

        # plot_overlay(img, mask, ground_truth, axis=2, slice=922)
        # plot_3d(mask)
