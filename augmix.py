def int_parameter(level, maxval):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.int_parameter(level, maxval)

def float_parameter(level, maxval):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.float_parameter(level, maxval)

def sample_level(n):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.sample_level(n)

def autocontrast(pil_img, _):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.autocontrast(pil_img, _)

def equalize(pil_img, _):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.equalize(pil_img, _)

def posterize(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.posterize(pil_img, level)

def rotate(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.rotate(pil_img, level)

def solarize(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.solarize(pil_img, level)

def shear_x(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.shear_x(pil_img, level)

def shear_y(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.shear_y(pil_img, level)

def translate_x(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.translate_x(pil_img, level)

def translate_y(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.translate_y(pil_img, level)

def color(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.color(pil_img, level)

def contrast(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.contrast(pil_img, level)

def brightness(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.brightness(pil_img, level)

def sharpness(pil_img, level):
    import TPT.data.augmix_ops 
    return TPT.data.augmix_ops.sharpness(pil_img, level)

def random_crop(img):
    from torchvision.transforms import RandomCrop
    size = 224
    return RandomCrop(size)(img)

def random_horizontal_flip(img):
    from torchvision.transforms import RandomHorizontalFlip
    return RandomHorizontalFlip()(img)

def random_resize_crop(img):
    from torchvision.transforms import RandomResizedCrop
    size = 224
    return RandomResizedCrop(size, scale=(0.8,1.0),  )(img)

def random_vertical_flip(img):
    from torchvision.transforms import RandomVerticalFlip
    return RandomVerticalFlip()(img)

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

post_augmentations = [
    random_crop, random_horizontal_flip, random_resize_crop
]

augmentations_basic = [
    random_crop, random_horizontal_flip, random_vertical_flip
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
