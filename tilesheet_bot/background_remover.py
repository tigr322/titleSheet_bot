from __future__ import annotations

from PIL import Image, ImageOps
from rembg import remove

from .interfaces import BackgroundRemoverInterface


def trim_transparent(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def has_transparency(img: Image.Image) -> bool:
    if img.mode in {"RGBA", "LA"}:
        alpha = img.getchannel("A")
        return alpha.getextrema()[0] < 255
    if img.mode == "P" and "transparency" in img.info:
        alpha = img.convert("RGBA").getchannel("A")
        return alpha.getextrema()[0] < 255
    return False


class RemoveBgBackgroundRemover(BackgroundRemoverInterface):
    def remove(self, img: Image.Image) -> Image.Image:
        img = ImageOps.exif_transpose(img)
        if has_transparency(img):
            return trim_transparent(img.convert("RGBA"))
        result = remove(img)
        return trim_transparent(result)
