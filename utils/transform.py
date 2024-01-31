import cv2


def resized_edge(img, scale, edge='short', interpolation='bicubic'):
    """Resize image to a proper size while keeping aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (int): The target size.
        edge (str): The edge to be matched. Options are "short", "long".
            Default: "short".
        interpolation (str): The interpolation method. Options are "nearest",
            "bilinear", "bicubic", "area", "lanczos". Default: "bicubic".

    Returns:
        ndarray: The resized image.
    """
    h, w = img.shape[:2]
    if edge == 'short':
        if h < w:
            ow = scale
            oh = int(scale * h / w)
        else:
            oh = scale
            ow = int(scale * w / h)
    elif edge == 'long':
        if h > w:
            ow = scale
            oh = int(scale * h / w)
        else:
            oh = scale
            ow = int(scale * w / h)
    else:
        raise ValueError(
            f'The edge must be "short" or "long", but got {edge}.')

    if interpolation == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif interpolation == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif interpolation == 'area':
        interpolation = cv2.INTER_AREA
    elif interpolation == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError(
            f'The interpolation must be "nearest", "bilinear", "bicubic", '
            f'"area" or "lanczos", but got {interpolation}.')

    img = cv2.resize(img, (ow, oh), interpolation=interpolation)

    return img

def center_crop(img, crop_size):
    """Crop the center of image.

    Args:
        img (ndarray): The input image.
        crop_size (int): The crop size.

    Returns:
        ndarray: The cropped image.
    """
    h, w = img.shape[:2]
    th, tw = crop_size, crop_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    img = img[i:i + th, j:j + tw, ...]
    return img