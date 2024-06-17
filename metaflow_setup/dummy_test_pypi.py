from metaflow import FlowSpec, step, batch, conda, conda_base

try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
    print("Env file found!")
except ImportError:
    print("Env file not found!")


@conda_base(python='3.11.9')
class dummy_test_pypi(FlowSpec):
    @batch(gpu=1, image="docker.io/tensorflow/tensorflow:latest-gpu", queue="job-queue-gpu-metaflow",)
    # @pypi(packages={'keras-cv': '0.9.0', 'tensorflow': '2.16.1'})
    @conda(libraries={'keras-cv': '0.9.0', 'tensorflow': '2.15', 'pycocotools': '2.0.6'})
    @step
    def start(self):

        # import tensorflow as tf
        # print("tensorflow" + tf.__version__)
        # import keras_cv
        # print("keras_cv" + keras_cv.__version__)

        import tensorflow as tf
        print("tensorflow" + tf.__version__)
        import keras_cv
        print("keras_cv:" + keras_cv.__version__)

        import sys
        print("Python version")
        print(sys.version)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.next(self.end)

    @step
    def end(self):

        print("All done. \n\n Congratulations!\n")
        return


if __name__ == '__main__':
    dummy_test_pypi()
