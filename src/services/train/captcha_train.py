"""
Created By: ishwor subedi
Date: 2024-03-28
 """

import tensorflow as tf

from src.services.train.architecture import train_model
from src.entity.configuration import ModelConfigs

try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
from mltu.annotations.images import CVImage

import os


class ModelTrainer:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.configs = ModelConfigs()
        self.vocab = set()
        self.max_len = 0

    def prepare_data(self, dataset_path):
        """
        Prepares the data for training and testing by reading the train_images and their labels from the dataset path.
        :param dataset_path: The path to the dataset.
        :return: A list of image paths and their corresponding labels.
        """
        dataset = []
        for file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file)
            label = os.path.splitext(file)[0]
            dataset.append([file_path, label])
            self.vocab.update(list(label))
            self.max_len = max(self.max_len, len(label))
        return dataset

    def train(self):
        """
        Trains the models in the custom made architecture and saves the models to the models path.and also prepare the
        data in csv format for the training and testing data and convert the .h5 mocdel into onnx format for the
        deployment purpose. :return:  None
        """
        train_data = self.prepare_data(self.train_dir)
        test_data = self.prepare_data(self.test_dir)

        # Save vocab and maximum text length to configs
        self.configs.vocab = "".join(self.vocab)
        self.configs.max_text_length = self.max_len
        self.configs.save()

        # Create a data provider for the training and testing data
        train_data_provider = DataProvider(
            dataset=train_data,
            skip_validation=True,
            batch_size=self.configs.batch_size,
            data_preprocessors=[ImageReader(CVImage)],
            transformers=[
                ImageResizer(self.configs.width, self.configs.height),
                LabelIndexer(self.configs.vocab),
                LabelPadding(max_word_length=self.configs.max_text_length, padding_value=len(self.configs.vocab))
            ],
        )
        test_data_provider = DataProvider(
            dataset=test_data,
            skip_validation=True,
            batch_size=self.configs.batch_size,
            data_preprocessors=[ImageReader(CVImage)],
            transformers=[
                ImageResizer(self.configs.width, self.configs.height),
                LabelIndexer(self.configs.vocab),
                LabelPadding(max_word_length=self.configs.max_text_length, padding_value=len(self.configs.vocab))
            ],
        )

        # Augment training data with random brightness, rotation and erode/dilate
        train_data_provider.augmentors = [RandomBrightness(), RandomRotate(), RandomErodeDilate()]

        # Creating TensorFlow models architecture
        model = train_model(
            input_dim=(self.configs.height, self.configs.width, 3),
            output_dim=len(self.configs.vocab),
        )

        # Compile the models and print summary
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.configs.learning_rate),
            loss=CTCloss(),
            metrics=[CWERMetric(padding_token=len(self.configs.vocab))],
            run_eagerly=False
        )
        model.summary(line_length=110)

        # Define path to save the models
        os.makedirs(self.configs.model_path, exist_ok=True)

        # Define callbacks
        earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min")
        checkpoint = ModelCheckpoint(f"{self.configs.model_path}/models.h5", monitor="val_CER", verbose=1,
                                     save_best_only=True, mode="min")
        trainLogger = TrainLogger(self.configs.model_path)
        tb_callback = TensorBoard(f"{self.configs.model_path}/logs", update_freq=1)
        reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1,
                                           mode="min")
        model2onnx = Model2onnx(f"{self.configs.model_path}/models.h5")

        # Train the models
        model.fit(
            train_data_provider,
            validation_data=test_data_provider,
            epochs=self.configs.train_epochs,
            callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
            workers=self.configs.train_workers

        )

        train_data_provider.to_csv(os.path.join(self.configs.model_path, "train.csv"))
        test_data_provider.to_csv(os.path.join(self.configs.model_path, "val.csv"))
