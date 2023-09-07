from cardscan.visualize_pipeline_app import run_app
from cardscan.pipeline_instances import card_images


def run():
    # Open an application that uses the camera as input and display every steps of the pipeline
    run_app(card_images)


if __name__ == "__main__":
    run()
