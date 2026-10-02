"""讓 `python -m vtuber` 可以執行（與安裝後的 `aivtuber` 指令相同）。"""

from vtuber.cli import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m vtuber"))
