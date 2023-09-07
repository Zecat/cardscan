import pygame
import sys
import cv2
import math
from cardscan.pipeline import Pipeline
from cardscan.image_processing import *
from typing import Tuple


def grid_layout_tile_size(tile_ratio, tile_count, window_size):
    window_w, window_h = window_size
    for cols in range(1, 30):
        new_tile_width = window_w // cols
        new_tile_height = int(new_tile_width / tile_ratio)
        rows = math.ceil(tile_count / cols)
        if rows * new_tile_height <= window_h:
            return (new_tile_width, new_tile_height), (rows, cols)


def cv2pygame(frame):
    """Create a pygame surface from opencv BVR or GRAY image. Applies RGB conversion and swap columns and rows."""
    return pygame.surfarray.make_surface(cv2.transpose(copy_rgb(frame)))


def scaleWidthKeepRatio(frame: MatLike, bounding_size: Tuple[int, int]) -> MatLike:
    bound_w, bound_h = bounding_size
    frame_h, frame_w, *_ = frame.shape
    scale_factor_fit_w = bound_w / frame_w
    scale_factor_fit_h = bound_h / frame_h
    scale_factor = min(scale_factor_fit_w, scale_factor_fit_h)
    resized_frame = cv2.resize(
        frame, (int(frame_w * scale_factor), int(frame_h * scale_factor))
    )
    return resized_frame


class Thumbnail:
    def __init__(self, cv2_img, resize, top_left=(0, 0), label: str = "image"):
        img = scaleWidthKeepRatio(cv2_img, resize)
        self.surface = cv2pygame(img)  # pygame.Surface
        self.rect = self.surface.get_rect()
        self.rect.topleft = top_left  # Set the initial position
        self.label = label

    def get_labeled_surface(self):
        # Create a Pygame font object
        font = pygame.font.Font(None, 24)  # None for default font, 36 for font size

        # Define the text and text color
        text_color = (255, 255, 255)  # White color (RGB)
        text_surface = font.render(self.label, True, text_color)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (20, 20)  # self.rect.topleft

        # Draw the text onto the screen
        self.surface.blit(text_surface, text_rect)

        return self.surface


def get_top_left(i: int, cols: int, size: Tuple[int, int]):
    col_i = i % cols
    row_i = i // cols
    return col_i * size[0], row_i * size[1]


def run_app(pipeline: Pipeline):
    pygame.init()

    # Get the current screen resolution (optional)
    screen_info = pygame.display.Info()
    (WINDOW_SIZE) = screen_info.current_w, screen_info.current_h

    BACKGROUND_COLOR = (255, 255, 255)

    # Create a Pygame window
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.FULLSCREEN)
    pygame.display.set_caption("Image Viewer")

    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    aspect_ratio = frame_width / frame_height

    selected_index = None
    running = True
    images = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if selected_index is None:
                        for i, image in enumerate(images):
                            if image.rect.collidepoint(event.pos):
                                selected_index = i
                    else:
                        selected_index = None

        ret, cam_frame = cam.read()
        if ret is None:
            break
        output, intermediate_results = pipeline.run_debug(cam_frame)
        final_frames = []
        for label, (_, debug_cb) in intermediate_results.items():
            if debug_cb is not None:
                final_frames += [(label, frame) for frame in debug_cb()]

        thumbnail_size, (tot_rows, tot_cols) = grid_layout_tile_size(
            aspect_ratio, len(final_frames), WINDOW_SIZE
        )

        images = [
            Thumbnail(
                frame,
                thumbnail_size,
                get_top_left(i, tot_cols, thumbnail_size),
                label=label,
            )
            for i, (label, frame) in enumerate(final_frames)
        ]

        screen.fill(BACKGROUND_COLOR)

        [
            screen.blit(image.get_labeled_surface(), image.rect.topleft)
            for image in images
        ]

        pygame.display.flip()

    cam.release()
    pygame.quit()
    sys.exit()
