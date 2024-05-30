from metaflow import FlowSpec, step, current
import os

# Loading environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except ImportError:
    print("No dotenv package")


class main_flow(FlowSpec):

    @step
    def start(self):
        """
        Start-up: check everything works or fail fast!
        """

        import kaggle as kg

        # Print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        # Ensure user has set the appropriate env variables
        assert os.environ['KAGGLE_USERNAME']
        assert os.environ['KAGGLE_KEY']

        try:
            kg.api.authenticate()
            print("Authentication to Kaggle successful!")
        except Exception as e:
            print(f"Authentication failed! Error: {e}")

        self.next(self.pull_data)

    @step
    def pull_data(self):
        import kaggle as kg

        print('Pulling data from Kaggle')
        try:
            file_path = '../data_raw/'
            kg.api.dataset_download_files(dataset="edizaguirre/plants-dataset",
                                          path=file_path,
                                          unzip=True)
            print(f"File download successful! Data is in {file_path}")
        except Exception as e:
            print(f"Download failed! Error: {e}")

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
