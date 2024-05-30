from metaflow import FlowSpec, step, current
import os

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

        # print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        # Ensure user has set the appropriate env variables
        assert os.environ['KAGGLE_USERNAME']
        assert os.environ['KAGGLE_KEY']

        self.next(self.end)

    @step
    def end(self):
        """
        Just say bye!
        """

        print("All done\n\nCongratulations!\n")
        return


if __name__ == '__main__':
    main_flow()
