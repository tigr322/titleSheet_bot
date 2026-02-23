from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class TileProcessor:
    width: int
    height: int
    fit_mode: str = "contain"

    def process(self, img: Image.Image) -> Image.Image:
        if self.fit_mode == "contain":
            return self._contain(img)
        if self.fit_mode == "cover":
            return self._cover(img)
        raise ValueError(f"Unsupported fit_mode: {self.fit_mode}")

    def _contain(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGBA")
        scale = min(self.width / img.width, self.height / img.height)
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        offset = ((self.width - new_w) // 2, (self.height - new_h) // 2)
        canvas.paste(resized, offset, resized)
        return canvas

    def _cover(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGBA")
        scale = max(self.width / img.width, self.height / img.height)
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = max(0, (new_w - self.width) // 2)
        top = max(0, (new_h - self.height) // 2)
        right = left + self.width
        bottom = top + self.height
        return resized.crop((left, top, right, bottom))
