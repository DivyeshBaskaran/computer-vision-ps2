import os
import cv2
import numpy as np

import ps2

INPUT_DIR = "input_images/"
OUTPUT_DIR = "output_images/"


def part_4a():
    img_bgr = cv2.imread(INPUT_DIR + 'dog.jpg', cv2.IMREAD_COLOR)

    thresholds = [0.1, 0.05, 0.001]
    output_labels = ['ps2-4-a-1', 'ps2-4-a-2', 'ps2-4-a-3']

    for keep, label in zip(thresholds, output_labels):
        img_compressed, compressed_frequency_img = ps2.compress_image_fft(
            img_bgr, keep)

        # Visualize the frequency domain image on a log scale so that the
        # dominant low frequencies don't wash out the rest of the spectrum.
        freq_vis = 20 * np.log(compressed_frequency_img + 1)
        freq_vis = np.clip(freq_vis, 0, 255)

        cv2.imwrite(os.path.join(OUTPUT_DIR, '{}.jpg'.format(label)),
                    img_compressed)
        cv2.imwrite(os.path.join(OUTPUT_DIR, '{}-freq.jpg'.format(label)),
                    freq_vis)
        print("Wrote {}.jpg and {}-freq.jpg (threshold={})".format(
            label, label, keep))


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    part_4a()
