import cv2
from skimage import morphology
from scipy.spatial import distance
import numpy as np
from typing import Tuple, List
import warnings


def name_removal(img):
    img_max = img.mean(axis=-1)
    h, w = img_max.shape
    h0, w0 = int(h/8), int(w/8)
    T = np.array([img_max[:h0, :w0].mean(), img_max[:h0, -w0:].mean(), 
                  img_max[-h0:, :w0].mean(), img_max[-h0:, -w0:].mean()])
    T = (T.sum() - T.max()) / 3
    vmax = np.max(img_max)
    BW = img_max <= (vmax+1)/25.0 + T
    th = [(vmax+1)/25.0, T]
    BW = morphology.remove_small_holes(BW, img_max.size/20.)
    BW = morphology.remove_small_objects(BW, 60)
    BW = np.invert(BW)
    img = img * BW[:, :, None]
    return img


def get_eye_region(img, stride=2):
    img_max = img[::stride, ::stride].mean(axis=-1)
    h, w = img_max.shape
    h0, w0 = int(h/4), int(w/4)
    T = (img_max[:h0, :w0].mean() + img_max[:h0, -w0:].mean() + 
        img_max[-h0:, :w0].mean() + img_max[-h0:, -w0:].mean()) / 4
    bw = morphology.remove_small_objects(img_max > T, w*h//100)
    bw = (bw * 255).astype(np.uint8)
    if bw.sum() < 10000:
        bw = (255 * np.ones(bw.shape)).astype(np.uint8)
    xy = np.where(bw)
    xmin, xmax = xy[0].min(), xy[0].max() + 1
    ymin, ymax = xy[1].min(), xy[1].max() + 1
    bw = cv2.resize(bw, (img.shape[1], img.shape[0]))
    return bw, [stride*xmin, stride*xmax, stride*ymin, stride*ymax]


def eye_region_detection(image: np.ndarray, stride: int = 8, contour_stride: int = 2,
                         mode: str = 'circle') -> Tuple[np.ndarray, List]:

    PI = 3.1415926535897932384626

    def get_ellipse_param(major_radius, minor_radius, angle):
        a, b = major_radius, minor_radius
        sin_theta = np.sin(-angle)
        cos_theta = np.cos(-angle)
        A = a**2 * sin_theta**2 + b**2 * cos_theta**2
        B = 2 * (a**2 - b**2) * sin_theta * cos_theta
        C = a**2 * cos_theta**2 + b**2 * sin_theta**2
        F = -a**2 * b**2
        return A, B, C, F

    def calculate_rectangle(A, B, C, F):
        y = np.sqrt(4*A*F / (B**2 - 4*A*C))
        y1, y2 = -np.abs(y), np.abs(y)
        x = np.sqrt(4*C*F / (B**2 - 4*C*A))
        x1, x2 = -np.abs(x), np.abs(x)
        return (x1, y1), (x2, y2)

    def get_rectangle(rrt):
        major_radius = rrt[1][0]/2
        minor_radius = rrt[1][1]/2
        angle = rrt[2]*PI*2/360
        center_x = rrt[0][0]
        center_y = rrt[0][1]
        A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
        p1, p2 = calculate_rectangle(A, B, C, F)
        return [center_y+p1[1], center_y+p2[1]+1, center_x+p1[0], center_x+p2[0]+1]

    assert isinstance(image, np.ndarray), "image has to be np.ndarray type"
    assert 1 <= stride <= 16, "stride has to be in [1, 16]"
    assert 1 <= contour_stride <= 8, "contour_stride has to be in [1, 8]"
    assert mode in ['circle', 'rectangle'], "mode has to be 'circle' or 'rectangle'"

#    image = image.astype(float)
    if not isinstance(stride, int):
        warnings.warn(f"stride={stride} is not int type, thus stride={int(stride)} is used.")
        stride = int(stride)
    if not isinstance(contour_stride, int):
        warnings.warn(f"contour_stride={contour_stride} is not int type, \
                      thus contour_stride={int(contour_stride)} is used.")
        contour_stride = int(contour_stride)

    if mode == 'circle':
        shift = 0
    else:
        shift = 0.5

    h_in, w_in = image.shape[:2]
    imgx = image[::stride, ::stride].max(axis=-1)
    h, w = imgx.shape
    h0, w0 = int(h/4), int(w/4)
    T = (imgx[:h0, :w0].mean() + imgx[:h0, -w0:].mean() +
         imgx[-h0:, :w0].mean() + imgx[-h0:, -w0:].mean()) / 4
    bw = morphology.remove_small_objects(imgx > T, w*h//100)
    if bw.sum() < 10000/(stride*stride/4):
        bw = np.ones((h_in, w_in), dtype=np.uint8)
        rect = [0, h_in, 0, w_in]
        return bw, rect

    bw = bw.astype(np.uint8)
    if int(cv2.__version__.split('.')[0]) < 4:
        _, contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    contours = np.vstack([cont[:, 0, :] for cont in contours])

    xy_label = np.min(contours, axis=1) == 0
    flag = (contours[:, 0] > 0) & (contours[:, 0] < w-1) & \
           (contours[:, 1] > 0) & (contours[:, 1] < h-1)
    contours = contours[flag, :]
    contours = contours[::contour_stride, :]

    dist_max = distance.cdist(contours, contours, 'euclidean').max(axis=1)

    pct50 = np.floor(np.percentile(dist_max, 70))
    pct75 = np.ceil(np.percentile(dist_max, 85))
    idx = np.where((dist_max > pct50) & (dist_max < pct75))
    contour_use = np.around(contours[idx, :]*stride).astype(int)

    rrt = cv2.fitEllipse(contour_use)
    if (min(rrt[1])/min(h_in, w_in) < 1/2) | (min(rrt[1])/min(h_in, w_in) > 1.5) | \
       (min(rrt[1])/max(rrt[1]) < 0.75):
        bw = np.ones((h_in, w_in), dtype=np.uint8)
        rect = [0, h_in, 0, w_in]
        return bw, rect

    center_shift_x = 0
    center_shift_y = 0
    diameter_shift = (stride-1)*shift*2
    if xy_label[0]:
        center_shift_x = (stride-1)*shift/2
        diameter_shift = (stride-1)*shift
    if xy_label[1]:
        center_shift_y = (stride-1)*shift/2
        diameter_shift = (stride-1)*shift
    rrt2 = ((rrt[0][0]+center_shift_x, rrt[0][1]+center_shift_y),
            (rrt[1][0]+diameter_shift, rrt[1][1]+diameter_shift), rrt[2])

    bw = np.zeros((h_in, w_in), dtype=np.uint8)
    cv2.ellipse(bw, rrt2, (1, 1, 1), -1)

    rect = get_rectangle(rrt2)
    rect = (rect + np.asarray([0, 1, 0, 1])).astype(int)
    rect[:2] = np.clip(rect[:2], 0, h_in)
    rect[2:] = np.clip(rect[2:], 0, w_in)
    rect = rect.tolist()
    return bw, rect
