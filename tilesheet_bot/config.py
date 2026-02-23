from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class BotConfig:
    token: str
    temp_dir: Path
    max_images_per_session: int
    read_timeout: float
    write_timeout: float
    connect_timeout: float
    pool_timeout: float
    debug_postprocess: bool
    debug_postprocess_dir: Path | None
    send_read_timeout: float
    send_write_timeout: float
    send_connect_timeout: float
    send_pool_timeout: float

    @classmethod
    def from_env(cls) -> "BotConfig":
        load_dotenv(Path(".env"))
        token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("BOT_TOKEN is not set")
        temp_dir = Path(os.getenv("TEMP_DIR", "temp"))
        max_images_raw = os.getenv("MAX_IMAGES_PER_SESSION", "0").strip()
        max_images = int(max_images_raw) if max_images_raw else 0
        if max_images < 0:
            raise RuntimeError("MAX_IMAGES_PER_SESSION must be 0 or positive")
        read_timeout = float(os.getenv("READ_TIMEOUT", "30").strip() or "30")
        write_timeout = float(os.getenv("WRITE_TIMEOUT", "30").strip() or "30")
        connect_timeout = float(os.getenv("CONNECT_TIMEOUT", "10").strip() or "10")
        pool_timeout = float(os.getenv("POOL_TIMEOUT", "5").strip() or "5")
        if read_timeout <= 0 or write_timeout <= 0 or connect_timeout <= 0 or pool_timeout <= 0:
            raise RuntimeError("Timeouts must be positive numbers")
        send_read_timeout = float(
            os.getenv("SEND_READ_TIMEOUT", str(read_timeout)).strip() or str(read_timeout)
        )
        send_write_timeout = float(
            os.getenv("SEND_WRITE_TIMEOUT", str(write_timeout)).strip() or str(write_timeout)
        )
        send_connect_timeout = float(
            os.getenv("SEND_CONNECT_TIMEOUT", str(connect_timeout)).strip() or str(connect_timeout)
        )
        send_pool_timeout = float(
            os.getenv("SEND_POOL_TIMEOUT", str(pool_timeout)).strip() or str(pool_timeout)
        )
        if (
            send_read_timeout <= 0
            or send_write_timeout <= 0
            or send_connect_timeout <= 0
            or send_pool_timeout <= 0
        ):
            raise RuntimeError("Send timeouts must be positive numbers")
        debug_postprocess = (
            os.getenv("DEBUG_POSTPROCESS", "0").strip().lower() in {"1", "true", "yes"}
        )
        debug_dir_raw = os.getenv("DEBUG_POSTPROCESS_DIR", "").strip()
        debug_postprocess_dir = Path(debug_dir_raw) if debug_dir_raw else None
        if debug_postprocess and debug_postprocess_dir is None:
            debug_postprocess_dir = temp_dir / "debug_postprocess"
        return cls(
            token=token,
            temp_dir=temp_dir,
            max_images_per_session=max_images,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
            connect_timeout=connect_timeout,
            pool_timeout=pool_timeout,
            debug_postprocess=debug_postprocess,
            debug_postprocess_dir=debug_postprocess_dir,
            send_read_timeout=send_read_timeout,
            send_write_timeout=send_write_timeout,
            send_connect_timeout=send_connect_timeout,
            send_pool_timeout=send_pool_timeout,
        )
