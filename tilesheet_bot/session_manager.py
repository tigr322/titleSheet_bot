from __future__ import annotations

from dataclasses import dataclass, field

from .models import StoredImage, TilesheetParams


@dataclass
class Session:
    params: TilesheetParams
    images: list[StoredImage] = field(default_factory=list)
    total_frames: int = 0

    @property
    def capacity(self) -> int:
        return self.params.grid_cols * self.params.grid_rows


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}

    def start(self, chat_id: int, params: TilesheetParams) -> Session:
        session = Session(params=params)
        self._sessions[chat_id] = session
        return session

    def get(self, chat_id: int) -> Session | None:
        return self._sessions.get(chat_id)

    def clear(self, chat_id: int) -> Session | None:
        return self._sessions.pop(chat_id, None)
