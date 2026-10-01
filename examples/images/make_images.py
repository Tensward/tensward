"""Draws the example images: an invoice, a table, a bar chart and an app screen, with text in a
5x7 bitmap font. They exist so the prompts in ../prompts-images.jsonl have real files to point
at; use your own images for real measurements. Standard library only: python3 make_images.py"""

import struct
import zlib
from pathlib import Path

WIDTH, HEIGHT = 640, 480
WHITE, INK, GREY, LINE = (255, 255, 255), (30, 34, 40), (150, 156, 165), (210, 214, 220)
BLUE, GREEN, ORANGE, RED = (46, 107, 214), (52, 160, 94), (232, 140, 40), (205, 70, 60)


# 5x7 glyphs, seven row bytes each, the leftmost pixel in bit 4.
GLYPHS = {
    " ": "00 00 00 00 00 00 00",
    "A": "0e 11 11 1f 11 11 11",
    "B": "1e 11 11 1e 11 11 1e",
    "C": "0e 11 10 10 10 11 0e",
    "D": "1e 11 11 11 11 11 1e",
    "E": "1f 10 10 1e 10 10 1f",
    "F": "1f 10 10 1e 10 10 10",
    "G": "0e 11 10 17 11 11 0f",
    "H": "11 11 11 1f 11 11 11",
    "I": "0e 04 04 04 04 04 0e",
    "J": "07 02 02 02 02 12 0c",
    "K": "11 12 14 18 14 12 11",
    "L": "10 10 10 10 10 10 1f",
    "M": "11 1b 15 15 11 11 11",
    "N": "11 11 19 15 13 11 11",
    "O": "0e 11 11 11 11 11 0e",
    "P": "1e 11 11 1e 10 10 10",
    "Q": "0e 11 11 11 15 12 0d",
    "R": "1e 11 11 1e 14 12 11",
    "S": "0f 10 10 0e 01 01 1e",
    "T": "1f 04 04 04 04 04 04",
    "U": "11 11 11 11 11 11 0e",
    "V": "11 11 11 11 11 0a 04",
    "W": "11 11 11 15 15 15 0a",
    "X": "11 11 0a 04 0a 11 11",
    "Y": "11 11 0a 04 04 04 04",
    "Z": "1f 01 02 04 08 10 1f",
    "0": "0e 11 13 15 19 11 0e",
    "1": "04 0c 04 04 04 04 0e",
    "2": "0e 11 01 02 04 08 1f",
    "3": "1f 02 04 02 01 11 0e",
    "4": "02 06 0a 12 1f 02 02",
    "5": "1f 10 1e 01 01 11 0e",
    "6": "06 08 10 1e 11 11 0e",
    "7": "1f 01 02 04 08 08 08",
    "8": "0e 11 11 0e 11 11 0e",
    "9": "0e 11 11 0f 01 02 0c",
    ".": "00 00 00 00 00 0c 0c",
    ",": "00 00 00 00 0c 04 08",
    ":": "00 0c 0c 00 0c 0c 00",
    "$": "04 0f 14 0e 05 1e 04",
    "-": "00 00 00 1f 00 00 00",
    "/": "01 01 02 04 08 10 10",
    "#": "0a 0a 1f 0a 1f 0a 0a",
    "%": "18 19 02 04 08 13 03",
}
FONT = {char: bytes.fromhex(rows) for char, rows in GLYPHS.items()}


class Canvas:
    def __init__(self, background: tuple[int, int, int] = WHITE) -> None:
        self.rows = [bytearray(bytes(background) * WIDTH) for _ in range(HEIGHT)]

    def rect(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        row = bytes(color) * width
        for line in self.rows[y : y + height]:
            line[3 * x : 3 * (x + width)] = row

    def text(
        self, x: int, y: int, words: str, color: tuple[int, int, int] = INK, scale: int = 2
    ) -> None:
        for index, char in enumerate(words):
            for row, bits in enumerate(FONT[char]):
                for column in range(5):
                    if bits >> (4 - column) & 1:
                        left = x + (6 * index + column) * scale
                        self.rect(left, y + row * scale, scale, scale, color)

    def png(self) -> bytes:
        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )


def invoice() -> Canvas:
    page = Canvas()
    page.rect(0, 0, WIDTH, 70, BLUE)
    page.text(40, 24, "NORTHWIND SUPPLY CO", WHITE, 3)
    page.text(40, 100, "12 DOCK ROAD")
    page.text(40, 124, "LEEDS LS1 4AB", GREY)
    page.text(360, 100, "INVOICE NO: INV-20418")
    page.text(360, 124, "DATE: 2026-09-12", GREY)
    page.text(360, 148, "DUE DATE: 2026-10-12", GREY)
    page.rect(40, 190, 560, 26, LINE)
    for left, heading in ((48, "DESCRIPTION"), (330, "QTY"), (410, "UNIT"), (512, "TOTAL")):
        page.text(left, 196, heading)
    items = (
        ("STEEL BRACKETS", "40", "$12.50", "$500.00"),
        ("HEX BOLTS M8 BOX", "10", "$8.40", "$84.00"),
        ("SAFETY GLOVES", "25", "$6.00", "$150.00"),
        ("WORK LAMPS", "6", "$32.00", "$192.00"),
        ("FREIGHT", "1", "$45.00", "$45.00"),
    )
    for index, (name, quantity, price, total) in enumerate(items):
        top = 230 + index * 32
        for left, value in ((48, name), (330, quantity), (410, price), (512, total)):
            page.text(left, top, value)
        page.rect(40, top + 22, 560, 1, LINE)
    page.rect(360, 410, 240, 40, GREEN)
    page.text(372, 423, "TOTAL DUE: $971.00", WHITE)
    return page


def table() -> Canvas:
    sheet = Canvas()
    columns = [40, 200, 340, 480, 600]
    rows = (
        ("MONTH", "ORDERS", "REVENUE", "RETURNS"),
        ("JAN", "1,204", "$48,300", "2.1%"),
        ("FEB", "1,310", "$52,900", "1.9%"),
        ("MAR", "1,455", "$58,700", "2.4%"),
        ("APR", "1,502", "$61,200", "2.2%"),
        ("MAY", "1,688", "$69,800", "1.8%"),
        ("JUN", "1,731", "$72,400", "2.0%"),
        ("JUL", "1,845", "$77,100", "1.7%"),
        ("AUG", "1,920", "$80,600", "1.6%"),
    )
    for index, cells in enumerate(rows):
        top = 40 + index * 44
        sheet.rect(40, top, 560, 40, INK if index == 0 else LINE if index % 2 else WHITE)
        for left, cell in zip(columns, cells):
            sheet.text(left + 12, top + 13, cell, WHITE if index == 0 else INK)
    for left in columns:
        sheet.rect(left, 40, 1, 396, GREY)
    return sheet


def chart() -> Canvas:
    plot = Canvas()
    plot.text(60, 10, "REVENUE BY QUARTER, $K")
    plot.rect(60, 40, 2, 380, INK)
    plot.rect(60, 418, 520, 2, INK)
    for level in range(1, 4):
        plot.rect(62, 418 - level * 100, 518, 1, LINE)
        plot.text(14, 411 - level * 100, str(level * 100), GREY)
    values = [120, 170, 150, 260, 310, 290, 350, 330]
    for index, value in enumerate(values):
        left = 84 + index * 62
        plot.rect(left, 418 - value, 40, value, BLUE if index < 4 else ORANGE)
        plot.text(left + 2, 400 - value, str(value))
        plot.text(left + 8, 428, f"Q{index % 4 + 1}")
    plot.text(150, 452, "2025")
    plot.text(398, 452, "2026")
    return plot


def screen() -> Canvas:
    app = Canvas((242, 244, 247))
    app.rect(0, 0, WIDTH, 56, INK)
    app.text(24, 21, "ORDERS DASHBOARD", WHITE)
    for left, label in ((290, "HOME"), (358, "ORDERS"), (446, "REPORTS"), (542, "HELP")):
        app.text(left, 21, label, GREY)
    app.rect(0, 56, 160, HEIGHT - 56, WHITE)
    labels = ("OVERVIEW", "ORDERS", "CUSTOMERS", "PRODUCTS", "INVOICES", "SUPPORT")
    for index, label in enumerate(labels):
        app.text(20, 86 + index * 44, label, BLUE if index == 1 else GREY)
    cards = (
        ("OPEN ORDERS", "1,920", BLUE),
        ("REVENUE", "$80.6K", GREEN),
        ("RETURNS", "1.6%", ORANGE),
    )
    for index, (label, value, color) in enumerate(cards):
        left = 190 + index * 150
        app.rect(left, 84, 130, 90, WHITE)
        app.text(left + 8, 96, label, GREY, 1)
        app.rect(left + 8, 120, 114, 36, color)
        app.text(left + 16, 130, value, WHITE)
    app.rect(190, 196, 430, 196, WHITE)
    app.text(206, 206, "RECENT ORDERS")
    orders = (
        ("#10482 ACME LTD", "$1,250.00"),
        ("#10481 BRIGHT CO", "$480.00"),
        ("#10480 CARTER INC", "$2,310.00"),
        ("#10479 DELTA LLC", "$96.50"),
        ("#10478 EVANS AND SON", "$730.00"),
    )
    for index, (name, amount) in enumerate(orders):
        top = 238 + index * 30
        app.text(206, top, name)
        app.text(500, top, amount)
    app.rect(290, 410, 160, 40, BLUE)
    app.text(302, 423, "SAVE CHANGES", WHITE)
    app.rect(460, 410, 160, 40, RED)
    app.text(472, 423, "DELETE ORDER", WHITE)
    return app


if __name__ == "__main__":
    here = Path(__file__).parent
    drawings = (("invoice", invoice), ("table", table), ("chart", chart), ("screen", screen))
    for name, draw in drawings:
        (here / f"{name}.png").write_bytes(draw().png())
