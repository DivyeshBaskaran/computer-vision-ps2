import os
import cv2

import ps2
from experiment import place_text

INPUT_DIR = "input_images/"
OUTPUT_DIR = "output_images/"


def template_match_test_fixed():
    input_images = ['scene_tl_1', 'scene_constr_1', 'waldo_1']
    output_labels = ['ps2-2-a-1', 'ps2-2-b-1', 'ps2-2-c-1']
    templates = [
        'tl_template.png', 'construction_template.png', 'waldo1_template.png'
    ]
    for img_fl, img_template_fl, label in zip(input_images, templates,
                                                output_labels):
        img_in = cv2.imread("input_images/{}.png".format(img_fl))
        img_template = cv2.imread("input_images/{}".format(img_template_fl))
        h, w = img_template.shape[:2]

        for method in ("tm_ssd", "tm_nssd", "tm_nccor"):
            top_left = ps2.template_match(img_in, img_template, method)
            bottom_right = (top_left[0] + w, top_left[1] + h)

            im_out = img_in.copy()
            cv2.rectangle(im_out, top_left, bottom_right, (255, 0, 255), 2)
            text = "(({}, {}))".format(top_left[0], top_left[1])
            place_text(text, top_left, im_out)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "{}-{}.png".format(method, label)),
                im_out)
            print("Wrote {}-{}.png (top_left={})".format(
                method, label, top_left))


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    template_match_test_fixed()
