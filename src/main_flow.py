from metaflow import FlowSpec, step, current, conda_base, conda
import os

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

        self.next(self.parse_records)

    # @conda(libraries={'tensorflow': '2.16.1', 'keras-cv': '0.9.0'})
    @step
    def parse_records(self):
        import tensorflow as tf
        from utils import parse_tfrecord_fn

        train_tfrecord_file = f'{self.file_path}leaves.tfrecord'
        val_tfrecord_file = f'{self.file_path}test_leaves.tfrecord'

        # Create a TFRecordDataset
        train_dataset = tf.data.TFRecordDataset([train_tfrecord_file])
        val_dataset = tf.data.TFRecordDataset([val_tfrecord_file])

        train_dataset = train_dataset.map(parse_tfrecord_fn)
        val_dataset = val_dataset.map(parse_tfrecord_fn)

        # Inspecting the data
        for data in train_dataset.take(1):
            print(data)

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
