from __future__ import annotations

from pathlib import Path
import uuid

from .interfaces import StorageInterface


class TempDirStorage(StorageInterface):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, filename_hint: str | None = None) -> Path:
        suffix = ""
        if filename_hint:
            suffix = Path(filename_hint).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".bin"
        name = f"{uuid.uuid4().hex}{suffix}"
        path = self.base_dir / name
        path.write_bytes(data)
        return path

    def cleanup(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
