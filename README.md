# Computer Vision — Problem Set 2: Template Matching and FFT

Solution for Problem Set 2 of Georgia Tech CS6476 (Computer Vision). The
assignment covers Hough-transform-based shape detection, template
matching, and the Fourier Transform (including image compression and
low-pass filtering in the frequency domain).

## Key files

| File | Description |
|---|---|
| [`ps2.py`](ps2.py) | Core implementation — all graded functions (see below) |
| [`experiment.py`](experiment.py) | Driver script that runs `ps2.py` against the input images and generates the output files used in the report |
| [`ps2_test.py`](ps2_test.py) | Local unit tests against the fixtures in [`tests/`](tests) |
| [`PS2_Report/PS2_Report.pdf`](PS2_Report/PS2_Report.pdf) | **Final writeup/report** with results, output images, and discussion answers |
| [`Instructions.pdf`](Instructions.pdf) | Original assignment instructions |
| [`FAQs.md`](FAQs.md) | Assignment FAQ (allowed/banned functions, grading notes, tuning tips) |
| `input_images/` | Source images (traffic light/construction-sign scenes, Waldo, templates) |
| `output_images/` | Generated output images referenced in the report |
| `generate_ps2_*.py` | Standalone scripts used to generate specific report figures |

### Functions implemented in `ps2.py`

- `traffic_light_detection` — locate a traffic light and its active color using Hough circle detection
- `construction_sign_detection` — locate a construction sign and return its centroid
- `template_match` — template matching via SSD, normalized SSD, and normalized correlation
- `dft` / `idft` — 1D Discrete Fourier Transform and its inverse
- `dft2` / `idft2` — 2D Discrete Fourier Transform and its inverse
- `compress_image_fft` — frequency-domain image compression by thresholding coefficients
- `low_pass_filter` — frequency-domain low-pass filtering with a circular mask

## Report

See [`PS2_Report/PS2_Report.pdf`](PS2_Report/PS2_Report.pdf) for the full
writeup, including output images and the discussion answers. The
`PS2_Report/` directory also contains the LaTeX source (`report.tex`,
`sections/`, `yourAnswers/`) used to build it.

## Running it

```bash
pip install numpy scipy opencv-python matplotlib
python experiment.py
python -m unittest ps2_test -v
```
