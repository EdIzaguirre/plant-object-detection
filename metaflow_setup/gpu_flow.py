from metaflow import FlowSpec, step, batch


class HelloGPUFlow(FlowSpec):

    @batch(
        gpu=1,
        image="docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
        queue="job-queue-gpu-metaflow",
    )
    @step
    def start(self):
        import torch

        if torch.cuda.is_available():
            print("GPU is available!")
        else:
            print("GPU is not available :(")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HelloGPUFlow()
