from cardscan.visualize_pipeline_app import run_app
from cardscan.pipeline_instances import (
    find_polygone_black_frame_pipeline as pipeline,
)


def run():
    # Open an application that uses the camera as input and display every steps of the pipeline
    run_app(pipeline)


if __name__ == "__main__":
    run()
