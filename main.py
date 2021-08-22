import numpy as np
import cv2
import glob

DEBUG = False
IMAGES_PATH = "./images/"

MASKS = {"red": {"upper": [170, 255, 255], "lower": [100, 150, 0]}}


def main(mask_name="red", debug=DEBUG):
    lower_color, upper_color = MASKS[mask_name]["lower"], MASKS[mask_name]["upper"]

    for image_path in glob.glob(f"{IMAGES_PATH}/*.tif"):
        image = cv2.imread(image_path)
        result = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array(lower_color)
        upper = np.array(upper_color)
        mask = cv2.inRange(image, lower, upper)

        mask_percentage = (mask > 0).mean()

        if debug is True:
            image_name = image_path.split("/")[2].replace(".tif", ".png")

            mask_image = cv2.bitwise_and(result, result, mask=mask)
            cv2.imwrite(f"./masks/{image_name}", mask_image)


if __name__ == '__main__':
    main(debug=True)
