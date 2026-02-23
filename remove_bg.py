from pathlib import Path
import argparse
from typing import Optional

from rembg import remove
from PIL import Image, ImageOps


def trim_transparent(img: Image.Image) -> Image.Image:
    """Trim fully transparent borders to fit the foreground."""
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


def process_image(input_path: str) -> Image.Image:
    inp = Path(input_path)
    with Image.open(inp) as img:
        img = ImageOps.exif_transpose(img)
        if has_transparency(img):
            return trim_transparent(img.convert("RGBA"))
        result = remove(img)
        return trim_transparent(result)


def load_image(input_path: str) -> Image.Image:
    inp = Path(input_path)
    with Image.open(inp) as img:
        return img.convert("RGBA")


def remove_background(
    input_path: str,
    output_path: str,
    width: Optional[int],
    height: Optional[int],
) -> None:
    img = process_image(input_path)
    save_output(img, output_path, width, height)


def stitch_images(img1: Image.Image, img2: Image.Image, direction: str) -> Image.Image:
    if direction == "horizontal":
        width = img1.width + img2.width
        height = max(img1.height, img2.height)
        out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        out.paste(img1, (0, 0), img1)
        out.paste(img2, (img1.width, 0), img2)
        return out

    width = max(img1.width, img2.width)
    height = img1.height + img2.height
    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    out.paste(img1, (0, 0), img1)
    out.paste(img2, (0, img1.height), img2)
    return out


def save_output(
    img: Image.Image,
    output_path: str,
    width: Optional[int],
    height: Optional[int],
) -> None:
    out = Path(output_path)
    if width is not None and height is not None:
        # Resize without preserving aspect ratio, as requested.
        img = img.resize((width, height), Image.LANCZOS)
    img.save(out, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove background and save PNG with transparency."
    )
    parser.add_argument("input_image", help="Path to input image (JPG/PNG)")
    parser.add_argument("output_png", help="Path to output PNG")
    parser.add_argument(
        "--second",
        help="Optional second input image to stitch with the first",
    )
    parser.add_argument(
        "--stitch",
        choices=["horizontal", "vertical"],
        help="Stitch direction when using --second (default: horizontal)",
    )
    parser.add_argument("--width", type=int, help="Output width in pixels")
    parser.add_argument("--height", type=int, help="Output height in pixels")
    args = parser.parse_args()

    if args.stitch is not None and args.second is None:
        parser.error("--stitch requires --second")
    if args.second is not None and args.stitch is None:
        args.stitch = "horizontal"

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be a positive integer")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be a positive integer")

    return args


def main() -> None:
    args = parse_args()
    if args.second:
        img1 = load_image(args.input_image)
        img2 = load_image(args.second)
        result = stitch_images(img1, img2, args.stitch)
    else:
        result = process_image(args.input_image)
    save_output(result, args.output_png, args.width, args.height)


if __name__ == "__main__":
    main()
