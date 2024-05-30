from metaflow import FlowSpec, step


class main_flow(FlowSpec):

    @step
    def start(self):
        """
        Start-up: check everything works or fail fast!
        """

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
