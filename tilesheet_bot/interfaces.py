from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image


class BackgroundRemoverInterface(Protocol):
    def remove(self, img: Image.Image) -> Image.Image:
        ...


class StorageInterface(Protocol):
    def save(self, data: bytes, filename_hint: str | None = None) -> Path:
        ...

    def cleanup(self, paths: list[Path]) -> None:
        ...
