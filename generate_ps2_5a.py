import os
import cv2
import numpy as np

import ps2

INPUT_DIR = "input_images/"
OUTPUT_DIR = "output_images/"


def part_5a():
    img_bgr = cv2.imread(INPUT_DIR + 'cat.jpg', cv2.IMREAD_COLOR)
    img_bgr = np.ndarray.astype(img_bgr, dtype=np.double)

    radii = [100, 50, 10]
    output_labels = ['ps2-5-a-1', 'ps2-5-a-2', 'ps2-5-a-3']

    for r, label in zip(radii, output_labels):
        img_low_pass, low_pass_frequency_img_mag = ps2.low_pass_filter(
            img_bgr, r)

        # Visualize the frequency domain image on a log scale so that the
        # dominant low frequencies don't wash out the rest of the spectrum.
        freq_vis = 20 * np.log(low_pass_frequency_img_mag + 1)
        freq_vis = np.clip(freq_vis, 0, 255)

        cv2.imwrite(os.path.join(OUTPUT_DIR, '{}.jpg'.format(label)),
                    img_low_pass)
        cv2.imwrite(os.path.join(OUTPUT_DIR, '{}-freq.jpg'.format(label)),
                    freq_vis)
        print("Wrote {}.jpg and {}-freq.jpg (radius={})".format(
            label, label, r))


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    part_5a()
