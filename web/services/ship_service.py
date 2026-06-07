"""业务逻辑服务层 — 封装数据库操作和 VLM 识别"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from config import load_config
from database import ShipDatabase
from tools import _vlm_infer

logger = logging.getLogger(__name__)


class ShipService:
    """船舶数据管理服务"""

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or load_config()
        self._db: ShipDatabase | None = None

    @property
    def db(self) -> ShipDatabase:
        if self._db is None:
            self._db = ShipDatabase(config=self._config)
        return self._db

    # —— CRUD ——

    def list_ships(self) -> list[dict]:
        data = self.db.source.load_all()
        return [{"hull_number": hn, "description": desc} for hn, desc in sorted(data.items())]

    def get_ship(self, hull_number: str) -> dict | None:
        desc = self.db.lookup(hull_number)
        if desc is None:
            return None
        return {"hull_number": hull_number, "description": desc}

    def create_ship(self, hull_number: str, description: str) -> bool:
        return self.db.add_ship(hull_number, description)

    def update_ship(self, hull_number: str, description: str) -> bool:
        return self.db.update_ship(hull_number, description)

    def delete_ship(self, hull_number: str) -> bool:
        return self.db.delete_ship(hull_number)

    def bulk_create(self, ships: dict[str, str]) -> dict:
        added = self.db.source.bulk_add(ships)
        if added > 0:
            self.db.reload()
        return {"added": added, "skipped": len(ships) - added}

    def search(self, keyword: str) -> list[dict]:
        return self.db.source.search_by_description(keyword)

    def stats(self) -> dict:
        source = self.db.source
        backend_type = "sqlite" if hasattr(source, "db_path") else "csv"
        return {"total_ships": source.count(), "backend": backend_type}

    # —— VLM 识别 ——

    def recognize_ship(self, image_bytes: bytes, filename: str) -> dict:
        """从图片识别船舶，返回 {hull_number, description, already_exists, existing_description}"""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = Path(filename).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp", ".gif": "image/gif",
        }
        mime = mime_map.get(ext, "image/jpeg")

        result = _vlm_infer(b64, prompt_mode="detailed", mime_type=mime)
        hull_number = result["hull_number"]
        description = result["description"]

        existing_desc = None
        if hull_number:
            existing_desc = self.db.lookup(hull_number)

        return {
            "hull_number": hull_number,
            "description": description,
            "already_exists": existing_desc is not None,
            "existing_description": existing_desc,
        }

    def recognize_and_add(self, image_bytes: bytes, filename: str) -> dict:
        """识别后自动入库"""
        result = self.recognize_ship(image_bytes, filename)
        hull_number = result["hull_number"]
        if not hull_number:
            return {"error": "未能识别出弦号，请手动输入", "result": result}
        action = self.db.upsert_ship(hull_number, result["description"])
        result["action"] = action
        return result