import cv2
import numpy as np
from matplotlib import pyplot as plt


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
 
    min_r = min(radii_range)
    max_r = max(radii_range)
 
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_r * 1.5,
        param1=30,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
 
    if circles is None:
        return (0, 0), 'red'
 
    circles = np.round(circles[0, :]).astype(int)
 
    # Group circles that share a similar x (vertically stacked, same
    # radius roughly) — find the best triplet of 3 circles.
    best_triplet = None
    best_score = None
    n = len(circles)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len({i, j, k}) != 3:
                    continue
                pts = sorted([circles[i], circles[j], circles[k]],
                             key=lambda c: c[1])
                xs = [p[0] for p in pts]
                rs = [p[2] for p in pts]
                x_spread = max(xs) - min(xs)
                r_spread = max(rs) - min(rs)
                # Vertical spacing should be roughly consistent
                gap1 = pts[1][1] - pts[0][1]
                gap2 = pts[2][1] - pts[1][1]
                gap_diff = abs(gap1 - gap2)
                score = x_spread + r_spread + gap_diff
                if gap1 <= 0 or gap2 <= 0:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_triplet = pts
 
    if best_triplet is None:
        c = circles[0]
        return (int(c[0]), int(c[1])), 'red'
 
    # top-to-bottom = red, yellow, green
    red_c, yellow_c, green_c = best_triplet
 
    colors = {
        'red': red_c,
        'yellow': yellow_c,
        'green': green_c,
    }
 
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    best_state = None
    best_sat_val = -1
    for name, c in colors.items():
        x, y, r = c
        x = int(np.clip(x, 0, img_in.shape[1] - 1))
        y = int(np.clip(y, 0, img_in.shape[0] - 1))
        h, s, v = img_hsv[y, x]
        # "on" light will be bright/saturated; "off" lights are dim
        brightness_score = int(s) + int(v)
        if brightness_score > best_sat_val:
            best_sat_val = brightness_score
            best_state = name
 
    center = (int(yellow_c[0]), int(yellow_c[1]))
    return center, best_state


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
 
    # Orange construction sign color range
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(img_hsv, lower_orange, upper_orange)
 
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0)
 
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return (cx, cy)


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and
    provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need
        to find the template
        img_template: numpy array representing template image which needs to be
        matched within the original image
        method: corresponds to one of the four metrics used to measure similarity
        between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with
        maximum match
    """
    """Each method is calls for a different metric to determine
       the degree to which the template matches the original image
       We are required to implement each technique using the
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        float,
    )
    top_left = []

    gray_orig = np.float64(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)) if img_orig.ndim == 3 else np.float64(img_orig)
    gray_template = np.float64(cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)) if img_template.ndim == 3 else np.float64(img_template)
    template_h, template_w = gray_template.shape

    """Once you have populated the result matrix with the similarity metric
    corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy
    post processing functions to extract location of maximum match"""
    # Sum of squared differences
    if method == "tm_ssd":
        out_h, out_w = result.shape
        for i in range(template_h):
            for j in range(template_w):
                diff = gray_orig[i:i+out_h, j:j+out_w] - gray_template[i, j]
                result += diff ** 2
        
        flat_index = np.argmin(result)
        out_y, out_x = np.unravel_index(flat_index, result.shape)
        top_left = (int(out_x), int(out_y))

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        for i, _row in enumerate(result):
            for j, _val in enumerate(result[i]):
                sub_img = gray_orig[i:i+template_h, j:j+template_w]
                diff = np.array(sub_img) - gray_template
                square = diff**2
                val = np.sum(square)
                result[i,j] = val

        result -= np.min(result)
        result /= np.max(result)
        result *= 255
        min_ind = np.argmin(result)
        cv2.imwrite('test.png', result)

        top_left = np.unravel_index(min_ind, result.shape)
        top_left = (top_left[1], top_left[0])
        return top_left

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        out_h, out_w = result.shape
        sum_I = np.zeros(result.shape)
        sum_I2 = np.zeros(result.shape)
        cross_corr = np.zeros(result.shape)
        for i in range(template_h):
            for j in range(template_w):
                window = gray_orig[i:i+out_h, j:j+out_w]
                sum_I += window
                sum_I2 += window ** 2
                cross_corr += window * gray_template[i, j]

        sum_T2 = np.sum(gray_template ** 2)
        energy_I = np.sqrt(sum_I2)
        energy_T = np.sqrt(sum_T2)

        result = cross_corr / (energy_I * energy_T + 1e-10)

        flat_index = np.argmax(result)
        out_y, out_x = np.unravel_index(flat_index, result.shape)
        top_left = (int(out_x), int(out_y))

    else:
        raise ValueError("Invalid method specified. Choose from 'tm_ssd', 'tm_nssd', or 'tm_nccor'.")

    return top_left


'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex_)

    n_samples = len(x)
    twiddle_factor = np.exp((-2j * np.pi)/n_samples)

    indices = np.arange(n_samples)
    dft_matrix = twiddle_factor**indices
    dft_matrix = dft_matrix**indices.reshape((n_samples, 1))

    y = np.dot(dft_matrix,x)
    return y


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    x = np.asarray(x, dtype=np.complex_)

    n_samples = len(x)
    twiddle_factor = np.exp((2j * np.pi)/n_samples)

    indices = np.arange(n_samples)
    idft_matrix = twiddle_factor**indices
    idft_matrix = idft_matrix**indices.reshape((n_samples, 1))

    y = np.dot(idft_matrix,x) * (1/n_samples)
    return y


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    img = np.asarray(img, dtype=np.complex_)
    height, width = img.shape

    twiddle_factor = np.exp((-2j * np.pi)/height)
    indices = np.arange(height)
    dft_matrix = twiddle_factor**indices
    dft_matrix = dft_matrix**indices.reshape((height, 1))
    intermediate = np.dot(dft_matrix,img)

    twiddle_factor = np.exp((-2j * np.pi)/width)
    indices = np.arange(width)
    dft_matrix = twiddle_factor**indices
    dft_matrix = dft_matrix**indices.reshape((width, 1))
    y = np.dot(dft_matrix,np.transpose(intermediate))

    y = np.transpose(y)
    return y


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    img = np.asarray(img, dtype=np.complex_)
    height, width = img.shape

    twiddle_factor = np.exp((2j * np.pi)/height)
    indices = np.arange(height)
    idft_matrix = twiddle_factor**indices
    idft_matrix = idft_matrix**indices.reshape((height, 1))
    intermediate = np.dot(idft_matrix,img) * (1/height)

    twiddle_factor = np.exp((2j * np.pi)/width)
    indices = np.arange(width)
    idft_matrix = twiddle_factor**indices
    idft_matrix = idft_matrix**indices.reshape((width, 1))
    y = np.dot(idft_matrix,np.transpose(intermediate)) * (1/width)

    y = np.transpose(y)
    return y


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image. (Make sure the data type of the np array is float64)
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain

    """
    height, width, channels = img_bgr.shape
    total_pixels = float(height) * float(width)
    img_compressed = np.asarray(img_bgr.copy(), dtype=np.float64)
    compressed_frequency_img = np.zeros((height, width, channels), dtype=np.float64)

    for channel in range(0, channels):
        freq = np.fft.fft2(img_bgr[:, :, channel])
        sorted_magnitudes = np.flip(np.sort(np.abs(freq.flatten())))

        index = int(np.floor(threshold_percentage * total_pixels))
        index = max(0, min(len(sorted_magnitudes) - 1, index))
        threshold = sorted_magnitudes[index]

        keep_mask = np.abs(freq) > threshold
        masked = np.zeros_like(freq, dtype=np.complex128)
        masked[keep_mask] = freq[keep_mask]

        compressed_frequency_img[:, :, channel] = np.abs(np.fft.fftshift(masked))
        compressed = np.fft.ifft2(masked)
        img_compressed[:, :, channel] = np.asarray(np.real(compressed), dtype=np.float64)

    return img_compressed, compressed_frequency_img


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image. (Make sure the data type of the np array is float64)
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """
    height, width, channels = img_bgr.shape
    img_low_pass = np.asarray(img_bgr.copy(), dtype=np.float64)
    low_pass_frequency_img = np.zeros((height, width, channels), dtype=np.float64)

    for channel in range(0, channels):
        freq = np.fft.fft2(img_bgr[:, :, channel])
        freq_shift = np.fft.fftshift(freq)

        radius = int(r)
        shape = np.shape(freq)
        mask = np.zeros(shape, dtype=np.float64)
        cv2.circle(mask, ((shape[1] // 2), (shape[0] // 2)), radius, 1, -1)

        freq_shift = np.multiply(freq_shift, mask)
        freq = np.fft.ifftshift(freq_shift)

        low_pass_frequency_img[:, :, channel] = np.abs(np.fft.fftshift(freq))
        low_pass = np.fft.ifft2(freq)
        img_low_pass[:, :, channel] = np.asarray(np.real(low_pass), dtype=np.float64)

    return img_low_pass, low_pass_frequency_img