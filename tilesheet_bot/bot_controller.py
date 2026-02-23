from __future__ import annotations

import io
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from rembg import remove
from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .background_remover import has_transparency
from .interfaces import BackgroundRemoverInterface, StorageInterface
from .models import StoredImage, TilesheetParams
from .session_manager import SessionManager
from .postprocess import FrameNormalizer, PostprocessConfig
from .tilesheet_builder import TilesheetBuilder


SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}
SUPPORTED_MIME = {"image/jpeg", "image/png"}


class ParseError(ValueError):
    pass


@dataclass
class SimpleImageJob:
    mode: str
    width: int | None = None
    height: int | None = None


def _parse_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ParseError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise ParseError(f"{label} must be positive")
    return parsed


def _parse_non_negative_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ParseError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ParseError(f"{label} must be 0 or positive")
    return parsed


def _parse_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ParseError(f"{label} must be a number") from exc
    return parsed


def _parse_clamp_range(value: str) -> tuple[float, float]:
    if ".." in value:
        left, right = value.split("..", 1)
    elif "," in value:
        left, right = value.split(",", 1)
    else:
        raise ParseError("clamp must be like 0.85..1.15")
    clamp_min = _parse_float(left.strip(), "clamp_min")
    clamp_max = _parse_float(right.strip(), "clamp_max")
    return clamp_min, clamp_max


def _parse_pair(token: str, label: str) -> tuple[int, int]:
    if "x" not in token:
        raise ParseError(f"{label} must be like 256x512")
    left, right = token.lower().split("x", 1)
    return _parse_int(left, label), _parse_int(right, label)


def parse_tilesheet_args(text: str) -> TilesheetParams:
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise ParseError("Unable to parse command arguments") from exc

    if not tokens:
        raise ParseError("Missing arguments")
    if tokens[0].startswith("/tilesheet") or tokens[0].lower() == "tilesheet":
        tokens = tokens[1:]

    if len(tokens) < 2:
        raise ParseError("Usage: /tilesheet 256x512 5x5 [padding] [order] [bg]")

    tile_width, tile_height = _parse_pair(tokens[0], "tile_size")
    grid_cols, grid_rows = _parse_pair(tokens[1], "grid")

    padding = 0
    order = "time"
    bg_mode = "remove"
    alpha_threshold = 16
    margin_bottom = 8
    clamp_min = 0.85
    clamp_max = 1.15
    resample = "bicubic"

    for token in tokens[2:]:
        low = token.lower()
        if "=" in low:
            key, value = low.split("=", 1)
            if key in {"padding", "pad"}:
                padding = _parse_non_negative_int(value, "padding")
            elif key == "order":
                order = value
            elif key in {"bg", "bg_mode"}:
                bg_mode = value
            elif key in {"alpha", "alpha_threshold"}:
                alpha_threshold = _parse_non_negative_int(value, "alpha_threshold")
            elif key in {"margin", "margin_bottom"}:
                margin_bottom = _parse_non_negative_int(value, "margin_bottom")
            elif key in {"clamp", "scale_clamp"}:
                clamp_min, clamp_max = _parse_clamp_range(value)
            elif key == "clamp_min":
                clamp_min = _parse_float(value, "clamp_min")
            elif key == "clamp_max":
                clamp_max = _parse_float(value, "clamp_max")
            elif key == "resample":
                resample = value
            else:
                raise ParseError(f"Unknown option: {token}")
            continue

        if low.isdigit():
            padding = _parse_non_negative_int(low, "padding")
            continue
        if low in {"time", "name"}:
            order = low
            continue
        if low in {"remove", "keep"}:
            bg_mode = low
            continue
        if low in {"nearest", "bicubic"}:
            resample = low
            continue
        raise ParseError(f"Unknown option: {token}")

    if order not in {"time", "name"}:
        raise ParseError("order must be 'time' or 'name'")
    if bg_mode not in {"remove", "keep"}:
        raise ParseError("bg_mode must be 'remove' or 'keep'")
    if alpha_threshold < 0 or alpha_threshold > 255:
        raise ParseError("alpha_threshold must be between 0 and 255")
    if clamp_min <= 0 or clamp_max <= 0 or clamp_min > clamp_max:
        raise ParseError("clamp range must be positive and min <= max")
    if resample.lower() not in {"nearest", "bicubic"}:
        raise ParseError("resample must be 'nearest' or 'bicubic'")

    return TilesheetParams(
        tile_width=tile_width,
        tile_height=tile_height,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        padding=padding,
        order=order,
        bg_mode=bg_mode,
        alpha_threshold=alpha_threshold,
        margin_bottom=margin_bottom,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        resample=resample.lower(),
    )


def parse_simple_image_args(
    text: str,
    command: str,
    require_size: bool,
) -> tuple[int | None, int | None]:
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise ParseError("Unable to parse command arguments") from exc

    if tokens and (tokens[0].startswith(f"/{command}") or tokens[0].lower() == command):
        tokens = tokens[1:]

    width = None
    height = None

    for token in tokens:
        low = token.lower()
        if "x" in low:
            if width is not None or height is not None:
                raise ParseError("size specified more than once")
            width, height = _parse_pair(low, "size")
            continue
        if low.startswith("width="):
            if width is not None:
                raise ParseError("width specified more than once")
            width = _parse_int(low.split("=", 1)[1], "width")
            continue
        if low.startswith("height="):
            if height is not None:
                raise ParseError("height specified more than once")
            height = _parse_int(low.split("=", 1)[1], "height")
            continue
        raise ParseError(f"Unknown option: {token}")

    if (width is None) != (height is None):
        raise ParseError("width and height must be provided together")
    if require_size and width is None:
        raise ParseError("Usage: /resize 512x512")

    return width, height


class BotController:
    def __init__(
        self,
        session_manager: SessionManager,
        background_remover: BackgroundRemoverInterface,
        storage: StorageInterface,
        max_images_per_session: int = 0,
        debug_postprocess_dir: Path | None = None,
        send_read_timeout: float | None = None,
        send_write_timeout: float | None = None,
        send_connect_timeout: float | None = None,
        send_pool_timeout: float | None = None,
    ) -> None:
        self._sessions = session_manager
        self._simple_jobs: dict[int, SimpleImageJob] = {}
        self._background_remover = background_remover
        self._storage = storage
        self._max_images_per_session = max_images_per_session
        self._debug_dir = debug_postprocess_dir
        self._send_read_timeout = send_read_timeout
        self._send_write_timeout = send_write_timeout
        self._send_connect_timeout = send_connect_timeout
        self._send_pool_timeout = send_pool_timeout

    def register(self, app: Application) -> None:
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("tilesheet", self.tilesheet))
        app.add_handler(CommandHandler("remove_bg", self.remove_bg))
        app.add_handler(CommandHandler("resize", self.resize))
        app.add_handler(CommandHandler("done", self.done))
        app.add_handler(CommandHandler("cancel", self.cancel))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        app.add_handler(MessageHandler(filters.Document.IMAGE, self.handle_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text(
                "Send /tilesheet 256x512 5x5 to start a session.\n"
                "Then upload images as photos or documents.\n"
                "Use /remove_bg [512x512] for a single image background removal.\n"
                "Use /resize 512x512 for a single image resize.\n"
                "Optional: alpha=16 margin=8 clamp=0.85..1.15 resample=bicubic.\n"
                "Use /done to finish early or /cancel to reset."
            )

    async def remove_bg(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_simple_job(update, "remove_bg", require_size=False)

    async def resize(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_simple_job(update, "resize", require_size=True)

    async def tilesheet(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        try:
            params = parse_tilesheet_args(update.message.text or "")
        except ParseError as exc:
            await update.message.reply_text(f"Invalid arguments: {exc}")
            return

        self._clear_simple_job(update.effective_chat.id)
        self._clear_session(update.effective_chat.id)
        session = self._sessions.start(update.effective_chat.id, params)
        await update.message.reply_text(
            f"Session started: {params.tile_width}x{params.tile_height}, "
            f"grid {params.grid_cols}x{params.grid_rows}, padding {params.padding}, "
            f"order {params.order}, bg {params.bg_mode}. "
            "Now send images."
        )

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        text = update.message.text.strip()
        low = text.lower()
        if low.startswith("tilesheet"):
            try:
                params = parse_tilesheet_args(text)
            except ParseError as exc:
                await update.message.reply_text(f"Invalid arguments: {exc}")
                return

            self._clear_simple_job(update.effective_chat.id)
            self._clear_session(update.effective_chat.id)
            self._sessions.start(update.effective_chat.id, params)
            await update.message.reply_text(
                f"Session started: {params.tile_width}x{params.tile_height}, "
                f"grid {params.grid_cols}x{params.grid_rows}, padding {params.padding}, "
                f"order {params.order}, bg {params.bg_mode}. "
                "Now send images."
            )
            return
        if low.startswith("remove_bg"):
            await self._start_simple_job(update, "remove_bg", require_size=False)
            return
        if low.startswith("resize"):
            await self._start_simple_job(update, "resize", require_size=True)
            return

    async def done(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        session = self._sessions.get(update.effective_chat.id)
        if not session:
            await update.message.reply_text("No active session. Use /tilesheet to start.")
            return
        if not session.images:
            await update.message.reply_text("No images yet. Send images or /cancel.")
            return
        await self._finalize_session(update, session)

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.effective_chat.id
        cleared_session = self._clear_session(chat_id)
        cleared_job = self._clear_simple_job(chat_id)
        if cleared_session or cleared_job:
            await update.message.reply_text("Canceled.")
        else:
            await update.message.reply_text("No active session.")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return
        chat_id = update.effective_chat.id
        session = self._sessions.get(chat_id)
        if not session:
            caption = update.message.caption or ""
            try:
                params = self._parse_tilesheet_from_text(caption)
            except ParseError as exc:
                await update.message.reply_text(f"Invalid arguments: {exc}")
                return
            if params:
                self._clear_simple_job(chat_id)
                session = self._sessions.start(chat_id, params)
                await update.message.reply_text(
                    f"Session started: {params.tile_width}x{params.tile_height}, "
                    f"grid {params.grid_cols}x{params.grid_rows}, padding {params.padding}, "
                    f"order {params.order}, bg {params.bg_mode}. "
                    "Processing image."
                )
            else:
                job = self._simple_jobs.pop(chat_id, None)
                if not job:
                    try:
                        job = self._parse_simple_job_from_text(caption)
                    except ParseError as exc:
                        await update.message.reply_text(f"Invalid arguments: {exc}")
                        return
                if job:
                    photo = update.message.photo[-1]
                    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
                    tg_file = await photo.get_file()
                    data = await tg_file.download_as_bytearray()
                    filename = f"photo_{photo.file_unique_id}.jpg"
                    await self._process_simple_job(update, bytes(data), filename, job)
                    return
                await update.message.reply_text("Start with /tilesheet, /remove_bg, or /resize.")
                return
        if session.total_frames >= session.capacity:
            await update.message.reply_text("Grid is full. Send /done or /cancel.")
            return
        if self._max_images_per_session and len(session.images) >= self._max_images_per_session:
            await update.message.reply_text("Session image limit reached. Send /done or /cancel.")
            return

        photo = update.message.photo[-1]
        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
        tg_file = await photo.get_file()
        data = await tg_file.download_as_bytearray()
        filename = f"photo_{photo.file_unique_id}.jpg"
        frame_count = self._count_frames_from_bytes(
            bytes(data), session.params.tile_width, session.params.tile_height
        )
        if session.total_frames + frame_count > session.capacity:
            await update.message.reply_text(
                "Too many frames for this grid. Send /done or increase grid size."
            )
            return
        self._add_image(session, bytes(data), filename, frame_count)
        await self._maybe_finalize(update, session)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document:
            return

        doc = update.message.document
        ext = ""
        if doc.file_name:
            ext = Path(doc.file_name).suffix.lower()
        mime_ok = doc.mime_type in SUPPORTED_MIME if doc.mime_type else False
        ext_ok = ext in SUPPORTED_EXT if ext else False
        if not (mime_ok or ext_ok):
            await update.message.reply_text("Unsupported file type. Send JPG or PNG.")
            return
        chat_id = update.effective_chat.id
        session = self._sessions.get(chat_id)
        if not session:
            caption = update.message.caption or ""
            try:
                params = self._parse_tilesheet_from_text(caption)
            except ParseError as exc:
                await update.message.reply_text(f"Invalid arguments: {exc}")
                return
            if params:
                self._clear_simple_job(chat_id)
                session = self._sessions.start(chat_id, params)
                await update.message.reply_text(
                    f"Session started: {params.tile_width}x{params.tile_height}, "
                    f"grid {params.grid_cols}x{params.grid_rows}, padding {params.padding}, "
                    f"order {params.order}, bg {params.bg_mode}. "
                    "Processing image."
                )
            else:
                job = self._simple_jobs.pop(chat_id, None)
                if not job:
                    try:
                        job = self._parse_simple_job_from_text(caption)
                    except ParseError as exc:
                        await update.message.reply_text(f"Invalid arguments: {exc}")
                        return
                if job:
                    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
                    tg_file = await doc.get_file()
                    data = await tg_file.download_as_bytearray()
                    await self._process_simple_job(
                        update, bytes(data), doc.file_name or "image", job
                    )
                    return
                await update.message.reply_text("Start with /tilesheet, /remove_bg, or /resize.")
                return
        if session.total_frames >= session.capacity:
            await update.message.reply_text("Grid is full. Send /done or /cancel.")
            return
        if self._max_images_per_session and len(session.images) >= self._max_images_per_session:
            await update.message.reply_text("Session image limit reached. Send /done or /cancel.")
            return

        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
        tg_file = await doc.get_file()
        data = await tg_file.download_as_bytearray()
        frame_count = self._count_frames_from_bytes(
            bytes(data), session.params.tile_width, session.params.tile_height
        )
        if session.total_frames + frame_count > session.capacity:
            await update.message.reply_text(
                "Too many frames for this grid. Send /done or increase grid size."
            )
            return
        self._add_image(session, bytes(data), doc.file_name or "image", frame_count)
        await self._maybe_finalize(update, session)

    def _add_image(self, session, data: bytes, filename: str, frame_count: int) -> None:
        path = self._storage.save(data, filename_hint=filename)
        stored = StoredImage(
            path=path,
            filename=filename,
            order_index=len(session.images),
            frame_count=frame_count,
        )
        session.images.append(stored)
        session.total_frames += frame_count

    async def _maybe_finalize(self, update: Update, session) -> None:
        if session.total_frames >= session.capacity:
            await self._finalize_session(update, session)

    async def _finalize_session(self, update: Update, session) -> None:
        try:
            await self._build_and_send(update, session)
        except Exception as exc:
            logging.exception("Failed to build tilesheet: %s", exc)
            if update.message:
                await update.message.reply_text("Failed to build tilesheet. Try again.")
        finally:
            self._clear_session(update.effective_chat.id)

    async def _build_and_send(self, update: Update, session) -> None:
        params = session.params
        ordered = self._ordered_images(session.images, params.order)
        frames = self._load_frames(ordered, params.tile_width, params.tile_height, params.bg_mode)
        if len(frames) > params.grid_cols * params.grid_rows:
            if update.message:
                await update.message.reply_text(
                    "Too many frames for this grid. Reduce images or increase grid size."
                )
            return
        normalizer = FrameNormalizer(
            PostprocessConfig(
                tile_width=params.tile_width,
                tile_height=params.tile_height,
                alpha_threshold=params.alpha_threshold,
                margin_bottom=params.margin_bottom,
                clamp_min=params.clamp_min,
                clamp_max=params.clamp_max,
                resample=params.resample,
                debug_dir=self._debug_dir,
            )
        )
        tiles = normalizer.normalize(frames)

        builder = TilesheetBuilder(
            tile_width=params.tile_width,
            tile_height=params.tile_height,
            grid_cols=params.grid_cols,
            grid_rows=params.grid_rows,
            padding=params.padding,
        )
        sheet = builder.build(tiles)

        out = io.BytesIO()
        sheet.save(out, format="PNG", optimize=True, compress_level=6)
        out.seek(0)

        caption = (
            f"tilesheet {params.tile_width}x{params.tile_height}, "
            f"grid {params.grid_cols}x{params.grid_rows}, frames {len(frames)}"
        )
        if update.message:
            await update.message.reply_document(
                document=InputFile(out, filename="tilesheet.png"),
                caption=caption,
                read_timeout=self._send_read_timeout,
                write_timeout=self._send_write_timeout,
                connect_timeout=self._send_connect_timeout,
                pool_timeout=self._send_pool_timeout,
            )

    async def _process_simple_job(
        self,
        update: Update,
        data: bytes,
        filename: str,
        job: SimpleImageJob,
    ) -> None:
        try:
            with Image.open(io.BytesIO(data)) as img:
                if job.mode == "remove_bg":
                    img = ImageOps.exif_transpose(img)
                    if has_transparency(img):
                        result = img.convert("RGBA")
                    else:
                        result = remove(img).convert("RGBA")
                else:
                    result = ImageOps.exif_transpose(img).convert("RGBA")
            if job.width is not None and job.height is not None:
                result = result.resize((job.width, job.height), Image.LANCZOS)
        except Exception as exc:
            logging.exception("Failed to process image: %s", exc)
            if update.message:
                await update.message.reply_text("Failed to process image. Try again.")
            return

        out = io.BytesIO()
        result.save(out, format="PNG", optimize=True, compress_level=6)
        out.seek(0)

        stem = Path(filename).stem if filename else "image"
        suffix = "nobg" if job.mode == "remove_bg" else "resized"
        out_name = f"{stem}_{suffix}.png"
        caption = self._simple_job_caption(job)
        if update.message:
            await update.message.reply_document(
                document=InputFile(out, filename=out_name),
                caption=caption,
                read_timeout=self._send_read_timeout,
                write_timeout=self._send_write_timeout,
                connect_timeout=self._send_connect_timeout,
                pool_timeout=self._send_pool_timeout,
            )

    def _ordered_images(self, images: list[StoredImage], order: str) -> list[StoredImage]:
        if order == "name":
            return sorted(images, key=lambda item: item.filename.lower())
        return sorted(images, key=lambda item: item.order_index)

    def _load_frames(
        self,
        images: list[StoredImage],
        tile_width: int,
        tile_height: int,
        bg_mode: str,
    ) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for item in images:
            with Image.open(item.path) as img:
                img = ImageOps.exif_transpose(img)
                for frame in self._extract_frames(img, tile_width, tile_height):
                    if bg_mode == "remove":
                        frames.append(self._background_remover.remove(frame))
                    else:
                        frames.append(frame.convert("RGBA"))
        return frames

    @staticmethod
    def _extract_frames(img: Image.Image, tile_width: int, tile_height: int) -> list[Image.Image]:
        width_div = img.width % tile_width == 0
        height_div = img.height % tile_height == 0
        if width_div and height_div:
            cols = img.width // tile_width
            rows = img.height // tile_height
            if cols > 1 or rows > 1:
                frames: list[Image.Image] = []
                for row in range(rows):
                    top = row * tile_height
                    for col in range(cols):
                        left = col * tile_width
                        frames.append(
                            img.crop((left, top, left + tile_width, top + tile_height))
                        )
                return frames
        if height_div and img.height > tile_height:
            count = img.height // tile_height
            frame_width = img.width
            return [
                img.crop((0, index * tile_height, frame_width, (index + 1) * tile_height))
                for index in range(count)
            ]
        if width_div and img.width > tile_width:
            count = img.width // tile_width
            frame_height = img.height
            return [
                img.crop((index * tile_width, 0, (index + 1) * tile_width, frame_height))
                for index in range(count)
            ]
        return [img]

    def _count_frames_from_bytes(self, data: bytes, tile_width: int, tile_height: int) -> int:
        try:
            with Image.open(io.BytesIO(data)) as img:
                return self._count_frames_for_image(img, tile_width, tile_height)
        except Exception:
            return 1

    @staticmethod
    def _count_frames_for_image(img: Image.Image, tile_width: int, tile_height: int) -> int:
        width_div = img.width % tile_width == 0
        height_div = img.height % tile_height == 0
        if width_div and height_div:
            cols = img.width // tile_width
            rows = img.height // tile_height
            if cols > 1 or rows > 1:
                return cols * rows
        if height_div and img.height > tile_height:
            return img.height // tile_height
        if width_div and img.width > tile_width:
            return img.width // tile_width
        return 1

    def _clear_session(self, chat_id: int) -> bool:
        session = self._sessions.clear(chat_id)
        if not session:
            return False
        self._storage.cleanup([item.path for item in session.images])
        return True

    def _clear_simple_job(self, chat_id: int) -> bool:
        return self._simple_jobs.pop(chat_id, None) is not None

    def _simple_job_caption(self, job: SimpleImageJob) -> str:
        if job.mode == "remove_bg":
            if job.width is not None and job.height is not None:
                return f"background removed, resized to {job.width}x{job.height}"
            return "background removed"
        return f"resized to {job.width}x{job.height}"

    def _parse_tilesheet_from_text(self, text: str) -> TilesheetParams | None:
        if not text:
            return None
        low = text.strip().lower()
        if low.startswith("/tilesheet") or low.startswith("tilesheet"):
            return parse_tilesheet_args(text)
        return None

    def _parse_simple_job_from_text(self, text: str) -> SimpleImageJob | None:
        if not text:
            return None
        low = text.strip().lower()
        if low.startswith("/remove_bg") or low.startswith("remove_bg"):
            width, height = parse_simple_image_args(text, "remove_bg", require_size=False)
            return SimpleImageJob(mode="remove_bg", width=width, height=height)
        if low.startswith("/resize") or low.startswith("resize"):
            width, height = parse_simple_image_args(text, "resize", require_size=True)
            return SimpleImageJob(mode="resize", width=width, height=height)
        return None

    async def _start_simple_job(
        self,
        update: Update,
        command: str,
        require_size: bool,
    ) -> None:
        if not update.message:
            return
        chat_id = update.effective_chat.id
        if self._sessions.get(chat_id):
            await update.message.reply_text(
                "Finish the tilesheet session first with /done or /cancel."
            )
            return
        try:
            width, height = parse_simple_image_args(
                update.message.text or "", command, require_size
            )
        except ParseError as exc:
            await update.message.reply_text(f"Invalid arguments: {exc}")
            return

        self._simple_jobs[chat_id] = SimpleImageJob(mode=command, width=width, height=height)
        if command == "resize":
            await update.message.reply_text(
                "Send an image (photo or PNG/JPG document) to resize."
            )
        else:
            await update.message.reply_text(
                "Send an image (photo or PNG/JPG document) to remove background."
            )
