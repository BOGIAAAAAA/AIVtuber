"""讓 `python -m vtuber` 可以執行（請在 aiVtuber/ 目錄下執行）。"""

from vtuber.cli import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m vtuber"))
