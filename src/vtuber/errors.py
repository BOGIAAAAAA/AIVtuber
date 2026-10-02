"""套件共用的例外類別。"""

from __future__ import annotations


class VTuberError(Exception):
    """可預期、使用者能自行修正的錯誤；CLI 只顯示訊息，不印 traceback。"""


class ConfigError(VTuberError):
    """設定檔格式或數值錯誤。"""


class TTSError(VTuberError):
    """語音合成失敗（連線錯誤、HTTP 錯誤或回應不是 WAV）。"""


class VTSError(VTuberError):
    """VTube Studio 連線或認證失敗。"""


class AudioDeviceError(VTuberError):
    """PyAudio 未安裝，或麥克風／喇叭裝置無法開啟。"""


class TrainingDataError(VTuberError):
    """訓練資料缺漏或不足以切出訓練樣本。"""


class ModelNotFoundError(VTuberError):
    """找不到微調後的模型目錄。"""
