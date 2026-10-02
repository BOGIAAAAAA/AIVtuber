"""VTube Studio：連線、以存檔的 token 認證、切換表情、關閉。

pyvts（0.3.3）的行為：request() 收 dict 並自行 json.dumps；
request_authenticate_token() 會先讀 token 檔，檔案沒有內容（或 force=True）才向 VTS 要新 token 並寫回檔案。
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any, Callable, Optional

from vtuber.config import VTSConfig
from vtuber.errors import VTSError

logger = logging.getLogger(__name__)

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"
EXPRESSION_SUFFIX = ".exp3.json"

ClientFactory = Callable[[VTSConfig], Any]


def expression_file_name(name: str) -> str:
    """VTS 要的是 .exp3.json 檔名；只給表情名稱時自動補上副檔名。"""
    name = name.strip()
    return name if name.lower().endswith(EXPRESSION_SUFFIX) else name + EXPRESSION_SUFFIX


def build_expression_request(expression: str, *, active: bool = True, fade_time: float = 0.25) -> dict[str, Any]:
    """組出啟用（或停用）表情的 ExpressionActivationRequest。"""
    return {
        "apiName": API_NAME,
        "apiVersion": API_VERSION,
        "requestID": "aivtuber-expression",
        "messageType": "ExpressionActivationRequest",
        "data": {"expressionFile": expression_file_name(expression), "fadeTime": fade_time, "active": active},
    }


class VTSController:
    """包裝 pyvts：負責連線、token 認證、表情切換與關閉。"""

    def __init__(self, config: VTSConfig, *, client_factory: Optional[ClientFactory] = None) -> None:
        self._config = config
        self._client_factory = client_factory or _create_pyvts_client
        self._client: Any = None

    async def connect(self) -> None:
        """連線並認證；失敗時丟出 VTSError，且不留下開著的連線。"""
        client = self._client_factory(self._config)
        url = f"ws://{self._config.host}:{self._config.port}"
        try:
            await client.connect()
        except Exception as exc:  # websockets 可能丟出 OSError、逾時或握手錯誤
            raise VTSError(
                f"Cannot connect to VTube Studio at {url} ({exc}); is VTube Studio running with the API enabled?"
            ) from exc
        self._client = client
        try:
            await self._authenticate()
        except BaseException:
            await self.close()
            raise

    async def set_expression(self, expression: str, active: bool = True) -> bool:
        """啟用或停用表情；VTS 回報錯誤（例如找不到表情檔）時回傳 False。

        啟用失敗記為警告；停用失敗通常只是該表情本來就沒啟用，只記 debug。
        """
        if self._client is None:
            raise VTSError("Not connected to VTube Studio")
        request = build_expression_request(expression, active=active, fade_time=self._config.fade_time)
        expression_file = request["data"]["expressionFile"]
        response = await self._client.request(request)
        if response.get("messageType") == "APIError":
            data = response.get("data") or {}
            report = logger.warning if active else logger.debug
            report(
                "VTube Studio rejected %s of expression %s: %s (errorID %s)",
                "activation" if active else "deactivation", expression_file, data.get("message"), data.get("errorID"),
            )
            return False
        logger.info("%s VTube Studio expression %s", "Activated" if active else "Deactivated", expression_file)
        return True

    async def switch_expression(self, expression: str, *, replaces: Iterable[str] = ()) -> bool:
        """先停用 replaces 裡的表情再啟用 expression，避免表情一直疊加；回傳是否啟用成功。"""
        target = expression_file_name(expression)
        for other in replaces:
            if expression_file_name(other) != target:
                await self.set_expression(other, active=False)
        return await self.set_expression(expression)

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def _authenticate(self) -> None:
        # pyvts 認證時會用 print() 輸出 plugin icon 與完整回應，改記到 debug log
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                await self._authenticate_with_saved_or_new_token()
        finally:
            if captured.getvalue().strip():
                logger.debug("pyvts output during authentication: %s", captured.getvalue().strip())

    async def _authenticate_with_saved_or_new_token(self) -> None:
        client = self._client
        if await client.read_token():
            if await _try_authenticate(client):
                logger.info("Authenticated with VTube Studio using the saved token")
                return
            logger.warning("Saved VTube Studio token was rejected; requesting a new one")
        logger.info("Requesting VTube Studio access: click 'Allow' in the VTube Studio popup")
        await client.request_authenticate_token(force=True)  # 取得新 token 並寫入 token_path
        if not await _try_authenticate(client):
            raise VTSError("VTube Studio authentication failed (was plugin access denied in VTube Studio?)")
        logger.info("Authenticated with VTube Studio; token saved to %s", self._config.token_path)


@contextlib.asynccontextmanager
async def vts_session(
    config: VTSConfig, *, client_factory: Optional[ClientFactory] = None
) -> AsyncIterator[Optional[VTSController]]:
    """連上 VTS，離開時（包括例外與 Ctrl+C）一定關閉；連不上時記錄錯誤並給 None，讓主流程照常進行。"""
    controller: Optional[VTSController] = VTSController(config, client_factory=client_factory)
    try:
        await controller.connect()
    except VTSError as exc:
        logger.error("%s; continuing without VTube Studio", exc)
        controller = None
    # 在 except 區塊外才 yield，主流程之後的例外才不會被串上這個 VTSError 當作 context
    if controller is None:
        yield None
        return
    try:
        yield controller
    finally:
        await controller.close()


async def _try_authenticate(client: Any) -> bool:
    try:
        return bool(await client.request_authenticate())
    except (KeyError, TypeError):
        # 回應是 APIError 時，pyvts 讀取 data["authenticated"] 會 KeyError
        return False


def _create_pyvts_client(config: VTSConfig) -> Any:
    import pyvts  # 延遲載入：pyvts 會連帶載入 OpenCV，只有真的用到 VTS 時才付這個成本

    config.token_path.parent.mkdir(parents=True, exist_ok=True)
    return pyvts.vts(
        plugin_info={
            "plugin_name": config.plugin_name,
            "developer": config.plugin_developer,
            "authentication_token_path": str(config.token_path),
        },
        vts_api_info={"version": API_VERSION, "name": API_NAME, "host": config.host, "port": config.port},
    )
