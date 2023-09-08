import cv2
from cardscan import scan


def run():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("cardscan", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("cardscan", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, cam_frame = cam.read()
        if not ret:
            break

        cam_frame = cv2.flip(cam_frame, 1)
        # Create an empty frame with the same dimensions
        output_frame = scan(cam_frame)
        if len(output_frame):
            cv2.imshow("cardscan", output_frame[0])

        key = cv2.waitKey(1)
        ESC_KEY = 27
        if key == ESC_KEY or key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return None


if __name__ == "__main__":
    run()
