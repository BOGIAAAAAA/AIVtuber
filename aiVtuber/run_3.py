"""舊版入口（相容用）。

所有功能已移到 vtuber 套件，這個檔案只把命令列參數轉交給 vtuber.cli，
讓 `python run_3.py --api`、`python run_3.py --train` 這類舊指令照樣能用。
新的寫法：在 aiVtuber/ 目錄下執行 `python -m vtuber ...`（參數完全相同）。
"""

from vtuber.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
