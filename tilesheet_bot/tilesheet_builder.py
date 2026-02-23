from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class TilesheetBuilder:
    tile_width: int
    tile_height: int
    grid_cols: int
    grid_rows: int
    padding: int = 0

    def build(self, tiles: list[Image.Image]) -> Image.Image:
        sheet_width = self.grid_cols * self.tile_width + (self.grid_cols - 1) * self.padding
        sheet_height = self.grid_rows * self.tile_height + (self.grid_rows - 1) * self.padding
        sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))

        for index, tile in enumerate(tiles):
            if index >= self.grid_cols * self.grid_rows:
                break
            row = index // self.grid_cols
            col = index % self.grid_cols
            x = col * (self.tile_width + self.padding)
            y = row * (self.tile_height + self.padding)
            sheet.paste(tile, (x, y), tile)

        return sheet
