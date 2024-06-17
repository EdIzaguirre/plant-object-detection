from metaflow import FlowSpec, step, current, batch, S3, conda, conda_base, environment, retry, pypi
from custom_decorators import pip
import os

# Loading environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except ImportError:
    print("Env file not found!")


# @conda_base(python='3.11.9')
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
        assert os.environ['WANDB_API_KEY']
        assert os.environ['WANDB_ENTITY']
        assert os.environ['WANDB_PROJECT']
        assert os.environ['S3_BUCKET_ADDRESS']

        self.next(self.train_model)

    @pip(libraries={'tensorflow': '2.16.1', 'keras-cv': '0.9.0', 'pycocotools': '2.0.7', 'wandb': '0.17.1'})
    # @batch(gpu=1, memory=8192, image="docker.io/tensorflow/tensorflow:latest-gpu", queue="job-queue-gpu-metaflow")
    # @batch(memory=15360, queue="job-queue-metaflow")
    # @environment(vars={
    #     "S3_BUCKET_ADDRESS": os.getenv('S3_BUCKET_ADDRESS'),
    #     'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
    #     'WANDB_PROJECT': os.getenv('WANDB_PROJECT'),
    #     'WANDB_ENTITY': os.getenv('WANDB_ENTITY')})
    @step
    def train_model(self):
        import tensorflow as tf

        # tf.config.optimizer.set_jit(False)
        # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

        from utils import parse_tfrecord_fn, dict_to_tuple, visualize_detections, class_mapping, create_model
        import keras
        import keras_cv
        import wandb
        from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

        assert os.getenv('WANDB_API_KEY')
        assert os.getenv('WANDB_ENTITY')
        assert os.getenv('WANDB_PROJECT')

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')

        file_path = 's3://' + os.getenv('S3_BUCKET_ADDRESS') + '/raw_data/'

        with S3() as s3:
            train_dataset_blob = s3.get(file_path + 'leaves.tfrecord').blob
            val_dataset_blob = s3.get(file_path + 'test_leaves.tfrecord').blob

        # Write the blobs to local files for TensorFlow to read
        train_tfrecord_file = 'train_leaves.tfrecord'
        val_tfrecord_file = 'val_test_leaves.tfrecord'

        with open(train_tfrecord_file, 'wb') as f:
            f.write(train_dataset_blob)

        with open(val_tfrecord_file, 'wb') as f:
            f.write(val_dataset_blob)

        # Create a TFRecordDataset
        train_dataset = tf.data.TFRecordDataset([train_tfrecord_file])
        val_dataset = tf.data.TFRecordDataset([val_tfrecord_file])

        # Iterate over a few entries and print their content. Uncomment this to look at the raw data
        # for record in train_dataset.take(1):
        #     example = tf.train.Example()
        #     example.ParseFromString(record.numpy())
        #     print(example)

        train_dataset = train_dataset.map(parse_tfrecord_fn)
        val_dataset = val_dataset.map(parse_tfrecord_fn)

        # Batching
        # Adding autotune for pre-fetching
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        IMG_SIZE = 416
        BBOX_FORMAT = "xyxy"

        # Start a run, tracking hyperparameters
        wandb.init(
            project=os.getenv('WANDB_PROJECT'),
            entity=os.getenv('WANDB_ENTITY'),
            config={
                "base_lr": 0.0001,
                "loss": "sparse_categorical_crossentropy",
                "epoch": 5,
                "batch_size": 16,
                "classification_loss": "focal",
                "box_loss": "smoothl1"
            }
        )

        config = wandb.config

        train_dataset = train_dataset.ragged_batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.ragged_batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)

        # Testing with only one batch
        train_dataset = train_dataset.take(1)
        val_dataset = val_dataset.take(1)

        print('Augmenting data')
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

        # Augmenting training set/resizing validation set
        train_dataset = train_dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

        # Converting data into tuples suitable for training
        train_dataset = train_dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

        # Including a global_clipnorm is extremely important in object detection tasks
        # optimizer_Adam = tf.keras.optimizers.Adam(
        optimizer_Adam = tf.keras.optimizers.legacy.Adam(
            learning_rate=config.base_lr,
            global_clipnorm=10.0
        )

        coco_metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format=BBOX_FORMAT, evaluate_freq=5
        )

        checkpoint_path = "best-custom-model.weights.h5"

        callbacks_list = [
            # Conducting early stopping to stop after 2 epochs of non-improving validation loss
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
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
            WandbMetricsLogger(log_freq=5),

            WandbModelCheckpoint("models")
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
            classification_loss=config.classification_loss,
            box_loss=config.box_loss,
            optimizer=optimizer_Adam,
            metrics=[coco_metrics],
            jit_compile=False
        )

        print('Beginning model fitting')
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epoch,
            callbacks=callbacks_list,
            verbose=1,
        )

        wandb.finish()

        self.model = {
            'model': model.to_json(),
            'model_weights': model.get_weights()
        }

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
