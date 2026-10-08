import os
import cv2

import ps2
from experiment import place_text

INPUT_DIR = "input_images/"
OUTPUT_DIR = "output_images/"


def part_2a():
    input_images = ['scene_tl_1', 'scene_constr_1', 'waldo_1']
    output_labels = ['ps2-2-a-1', 'ps2-2-b-1', 'ps2-2-c-1']
    templates = [
        'tl_template.png', 'construction_template.png', 'waldo1_template.png'
    ]

    # tm_ssd (and tm_nssd, which gave identical results) was the most
    # consistently accurate method across all three scenes, so that's what
    # we display in the report per the "pick the best" instruction.
    method = 'tm_ssd'

    for img_fl, img_template_fl, label in zip(input_images, templates,
                                                output_labels):
        img_in = cv2.imread("input_images/{}.png".format(img_fl))
        img_template = cv2.imread("input_images/{}".format(img_template_fl))

        top_left = ps2.template_match(img_in, img_template, method)
        h, w = img_template.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)

        im_out = img_in.copy()
        cv2.rectangle(im_out, top_left, bottom_right, (255, 0, 255), 2)
        text = "(({}, {}))".format(top_left[0], top_left[1])
        place_text(text, top_left, im_out)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "{}.jpg".format(label)), im_out)
        print("Wrote {}.jpg (method={}, top_left={})".format(
            label, method, top_left))


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    part_2a()
