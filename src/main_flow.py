from metaflow import FlowSpec, step, current, conda_base, conda, batch
import os
from custom_decorators import pip

# Loading environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except ImportError:
    print("Env file not found!")


# @conda_base(python='3.12.3')
class main_flow(FlowSpec):

    @step
    def start(self):
        """
        Start-up: check everything works or fail fast!
        """

        # Print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        # Ensure user has set the appropriate env variables
        assert os.environ['KAGGLE_USERNAME']
        assert os.environ['KAGGLE_KEY']

        self.next(self.pull_data)

    # @conda(libraries={'kaggle': '1.6.14'})
    @step
    def pull_data(self):
        import tensorflow as tf
        import kaggle as kg

        print('Pulling data from Kaggle')

        # Checking authentication
        try:
            kg.api.authenticate()
            print("Authentication to Kaggle successful!")
        except Exception as e:
            print(f"Authentication failed! Error: {e}")

        # Attempting download
        try:
            self.file_path = '../data_raw/'
            kg.api.dataset_download_files(dataset="edizaguirre/plants-dataset",
                                          path=self.file_path,
                                          unzip=True)
            print(f"File download successful! Data is in {self.file_path}")
        except Exception as e:
            print(f"Download failed! Error: {e}")

        train_tfrecord_file = f'{self.file_path}leaves.tfrecord'
        val_tfrecord_file = f'{self.file_path}test_leaves.tfrecord'

        # Create a TFRecordDataset
        train_dataset = tf.data.TFRecordDataset([train_tfrecord_file])
        val_dataset = tf.data.TFRecordDataset([val_tfrecord_file])

        self.next(self.parse_and_transform_records)

    # @conda(libraries={'tensorflow': '2.16.1', 'keras-cv': '0.9.0'})
    @batch(gpu=1,
           image="docker.io/tensorflow/tensorflow:latest-gpu",
           queue="job-queue-gpu-metaflow-v2",
           )
    @pip(libraries={'keras-cv': '0.9.0'})
    @step
    def parse_and_transform_records(self):
        import tensorflow as tf
        from utils import parse_tfrecord_fn, dict_to_tuple, visualize_detections, class_mapping, create_model
        import keras
        import keras_cv

        print('Parsing raw data and augmenting images')

        train_dataset = train_dataset.map(parse_tfrecord_fn)
        val_dataset = val_dataset.map(parse_tfrecord_fn)

        # Batching
        BATCH_SIZE = 32
        # Adding autotune for pre-fetching
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # Other constants
        NUM_ROWS = 4
        NUM_COLS = 8
        IMG_SIZE = 416
        BBOX_FORMAT = "xyxy"

        train_dataset = train_dataset.ragged_batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.ragged_batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

        # Defining augmentations
        augmenter = keras.Sequential(
            [
                keras_cv.layers.JitteredResize(
                    target_size=(IMG_SIZE, IMG_SIZE), scale_factor=(0.8, 1.25), bounding_box_format=BBOX_FORMAT
                ),
                keras_cv.layers.RandomFlip(mode="horizontal_and_vertical", bounding_box_format=BBOX_FORMAT),
                keras_cv.layers.RandomRotation(factor=0.06, bounding_box_format=BBOX_FORMAT),
                keras_cv.layers.RandomSaturation(factor=(0.4, 0.6)),
                keras_cv.layers.RandomHue(factor=0.2, value_range=[0, 255]),
            ]
        )

        # Resize and pad images
        inference_resizing = keras_cv.layers.Resizing(
            IMG_SIZE, IMG_SIZE, pad_to_aspect_ratio=True, bounding_box_format=BBOX_FORMAT
        )

        # Creating artifacts to visualize in notebook
        train_dataset = train_dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

        # Converting data into tuples suitable for training
        train_dataset = train_dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

        base_lr = 0.0001
        # including a global_clipnorm is extremely important in object detection tasks
        optimizer_Adam = tf.keras.optimizers.Adam(
            learning_rate=base_lr,
            global_clipnorm=10.0
        )

        coco_metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format=BBOX_FORMAT, evaluate_freq=5
        )

        class VisualizeDetections(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if (epoch + 1) % 5 == 0:
                    visualize_detections(
                        self.model, bounding_box_format=BBOX_FORMAT, dataset=val_dataset, rows=NUM_ROWS, cols=NUM_COLS
                    )

        checkpoint_path = "best-custom-model.weights.h5"

        callbacks_list = [
            # Conducting early stopping to stop after 6 epochs of non-improving validation loss
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
            ),

            # Saving the best model
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True
            ),

            # Custom metrics printing after each epoch
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                print(f"\nEpoch #{epoch + 1} \n" +
                      f"Loss: {logs['loss']:.4f} \n" +
                      f"mAP: {logs['MaP']:.4f} \n" +
                      f"Validation Loss: {logs['val_loss']:.4f} \n" +
                      f"Validation mAP: {logs['val_MaP']:.4f} \n")
            ),

            # Visualizing results after each n epoch
            VisualizeDetections()
        ]

        model = create_model(format=BBOX_FORMAT)

        # Customizing non-max supression of model prediction.
        model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
            bounding_box_format=BBOX_FORMAT,
            from_logits=True,
            iou_threshold=0.5,
            confidence_threshold=0.5,
        )

        # Using focal classification loss and smoothl1 box loss with coco metrics
        model.compile(
            classification_loss="focal",
            box_loss="smoothl1",
            optimizer=optimizer_Adam,
            metrics=[coco_metrics]
        )

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=40,
            callbacks=callbacks_list,
            verbose=1,
        )

        self.next(self.end)

    @step
    def end(self):
        """
        Just say bye!
        """

        print("All done. \n\n Congratulations!\n")
        return


if __name__ == '__main__':
    main_flow()
