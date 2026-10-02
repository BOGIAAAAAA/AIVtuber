from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Optional

import pytest

from vtuber.config import VTSConfig
from vtuber.errors import VTSError
from vtuber.vts import VTSController, build_expression_request, expression_file_name, vts_session


class FakeVTS:
    """照 pyvts 0.3.3 的行為模擬 pyvts.vts 與 VTube Studio：

    - read_token()：token 檔不存在時回傳目前的 token，存在時讀檔。
    - request_authenticate_token(force)：先讀 token 檔，只有讀到空值或 force=True 才跳授權視窗並寫檔。
    - request_authenticate()：讀取回應的 data["authenticated"]；回應是 APIError 時因此 KeyError。
    - 認證過程會用 print() 輸出 plugin icon 與失敗的回應。
    """

    def __init__(
        self,
        saved_token: Optional[str] = None,
        valid_tokens: tuple[str, ...] = ("issued-token",),
        api_error_tokens: tuple[str, ...] = (),
        user_allows: bool = True,
        connect_error: Optional[Exception] = None,
        missing_expressions: tuple[str, ...] = (),
        reject_deactivation: bool = False,
    ) -> None:
        self.saved_token = saved_token  # token 檔內容；None 表示檔案不存在
        self.valid_tokens = valid_tokens
        self.api_error_tokens = api_error_tokens
        self.user_allows = user_allows
        self.connect_error = connect_error
        self.missing_expressions = missing_expressions
        self.reject_deactivation = reject_deactivation
        self.authentic_token: Optional[str] = None
        self.popups = 0
        self.sent: list = []
        self.closed = False

    async def connect(self) -> None:
        if self.connect_error:
            raise self.connect_error

    async def read_token(self) -> Optional[str]:
        if self.saved_token is None:
            return self.authentic_token
        self.authentic_token = self.saved_token
        return self.authentic_token

    async def request_authenticate_token(self, force: bool = False) -> None:
        print(None)  # pyvts 會印出 plugin icon
        token = await self.read_token()
        if token is None or token == "" or force:
            self.popups += 1
            if self.user_allows:
                self.authentic_token = self.saved_token = "issued-token"
            else:
                print("authentication failed")

    async def request_authenticate(self) -> bool:
        response = self._authentication_response(self.authentic_token)
        authenticated = response["data"]["authenticated"]  # APIError 回應沒有這個欄位
        if not authenticated:
            print(response)
        return authenticated

    def _authentication_response(self, token: Optional[str]) -> dict:
        if not token or token in self.api_error_tokens:
            return {"messageType": "APIError", "data": {"errorID": 8, "message": "Invalid authentication request"}}
        return {"messageType": "AuthenticationResponse", "data": {"authenticated": token in self.valid_tokens}}

    async def request(self, message: dict) -> dict:
        self.sent.append(message)
        data = message["data"]
        if data["expressionFile"] in self.missing_expressions or (self.reject_deactivation and not data["active"]):
            return {"messageType": "APIError", "data": {"errorID": 452, "message": "Expression not found"}}
        return {"messageType": "ExpressionActivationResponse", "data": {}}

    async def close(self) -> None:
        self.closed = True


def _connect(fake: FakeVTS, config: Optional[VTSConfig] = None) -> VTSController:
    controller = VTSController(config or VTSConfig(), client_factory=lambda _config: fake)
    asyncio.run(controller.connect())
    return controller


def _sent_expressions(fake: FakeVTS) -> list[tuple[str, bool]]:
    return [(message["data"]["expressionFile"], message["data"]["active"]) for message in fake.sent]


def test_expression_request_matches_vts_public_api_format():
    request = build_expression_request("Happy", fade_time=0.5)

    assert request["apiName"] == "VTubeStudioPublicAPI"
    assert request["apiVersion"] == "1.0"
    assert request["messageType"] == "ExpressionActivationRequest"  # 不是查詢狀態的 ExpressionStateRequest
    assert request["data"] == {"expressionFile": "Happy.exp3.json", "fadeTime": 0.5, "active": True}
    assert isinstance(request["requestID"], str) and 0 < len(request["requestID"]) <= 64
    assert json.loads(json.dumps(request)) == request  # 可直接序列化成 JSON 物件


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Happy", "Happy.exp3.json"), ("Sad.exp3.json", "Sad.exp3.json"), (" Smile ", "Smile.exp3.json")],
)
def test_expression_file_name_adds_suffix_only_when_missing(name, expected):
    assert expression_file_name(name) == expected


def test_set_expression_sends_a_dict_to_pyvts_request():
    fake = FakeVTS(saved_token="issued-token")
    controller = _connect(fake, replace(VTSConfig(), fade_time=0.3))

    ok = asyncio.run(controller.set_expression("Sad"))

    assert ok is True
    (message,) = fake.sent
    assert isinstance(message, dict)  # pyvts.request() 會自己 json.dumps，傳字串會被重複編碼
    assert message["messageType"] == "ExpressionActivationRequest"
    assert message["data"] == {"expressionFile": "Sad.exp3.json", "fadeTime": 0.3, "active": True}


def test_api_error_response_is_reported_not_raised(caplog):
    controller = _connect(FakeVTS(saved_token="issued-token", missing_expressions=("Missing.exp3.json",)))

    with caplog.at_level(logging.WARNING, logger="vtuber.vts"):
        assert asyncio.run(controller.set_expression("Missing")) is False

    assert "Expression not found" in caplog.text


def test_switch_expression_deactivates_the_other_expression_before_activating():
    fake = FakeVTS(saved_token="issued-token")
    controller = _connect(fake)

    assert asyncio.run(controller.switch_expression("Sad", replaces=["Happy"])) is True

    assert _sent_expressions(fake) == [("Happy.exp3.json", False), ("Sad.exp3.json", True)]


def test_switch_expression_ignores_errors_when_the_other_expression_was_not_active(caplog):
    fake = FakeVTS(saved_token="issued-token", reject_deactivation=True)
    controller = _connect(fake)

    with caplog.at_level(logging.WARNING, logger="vtuber.vts"):
        assert asyncio.run(controller.switch_expression("Happy", replaces=["Sad"])) is True

    assert _sent_expressions(fake) == [("Sad.exp3.json", False), ("Happy.exp3.json", True)]
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]  # 停用失敗只記 debug


def test_switch_expression_reports_failure_when_the_target_is_missing():
    fake = FakeVTS(saved_token="issued-token", missing_expressions=("Angry.exp3.json",))
    controller = _connect(fake)

    assert asyncio.run(controller.switch_expression("Angry", replaces=["Happy"])) is False


def test_switch_expression_never_deactivates_the_target_itself():
    fake = FakeVTS(saved_token="issued-token")
    controller = _connect(fake)

    asyncio.run(controller.switch_expression("Happy", replaces=["Happy.exp3.json"]))

    assert _sent_expressions(fake) == [("Happy.exp3.json", True)]


def test_saved_token_is_reused_without_a_new_popup():
    fake = FakeVTS(saved_token="issued-token")

    _connect(fake)

    assert fake.popups == 0


def test_first_run_requests_token_once_and_persists_it():
    fake = FakeVTS(saved_token=None)

    _connect(fake)

    assert fake.popups == 1
    assert fake.saved_token == "issued-token"


def test_rejected_saved_token_forces_exactly_one_new_request():
    # pyvts 只在 token 檔是空的或 force=True 時才重新申請；舊 token 被撤銷時必須用 force
    fake = FakeVTS(saved_token="revoked-token")

    _connect(fake)

    assert fake.popups == 1
    assert fake.saved_token == "issued-token"


def test_api_error_authentication_response_is_treated_as_rejection():
    # 回應是 APIError 時 pyvts 會 KeyError；應視為認證失敗並重新申請，而不是整個崩潰
    fake = FakeVTS(saved_token="garbled-token", api_error_tokens=("garbled-token",))

    _connect(fake)

    assert fake.popups == 1


def test_denied_access_raises_and_closes_connection():
    fake = FakeVTS(saved_token=None, user_allows=False)

    with pytest.raises(VTSError, match="authentication failed"):
        _connect(fake)

    assert fake.closed


def test_pyvts_prints_during_authentication_stay_out_of_stdout(capsys):
    _connect(FakeVTS(saved_token="revoked-token"))

    assert capsys.readouterr().out == ""


def test_unreachable_vts_raises_vts_error_with_url():
    fake = FakeVTS(connect_error=ConnectionRefusedError("refused"))

    with pytest.raises(VTSError, match="ws://localhost:8001"):
        _connect(fake)


def test_vts_session_always_closes_even_when_the_body_fails():
    fake = FakeVTS(saved_token="issued-token")

    async def run() -> None:
        async with vts_session(VTSConfig(), client_factory=lambda _config: fake) as vts:
            assert vts is not None
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(run())

    assert fake.closed


def test_vts_session_yields_none_when_vts_is_unavailable(caplog):
    fake = FakeVTS(connect_error=OSError("connection refused"))

    async def run() -> object:
        async with vts_session(VTSConfig(), client_factory=lambda _config: fake) as vts:
            return vts

    with caplog.at_level(logging.ERROR, logger="vtuber.vts"):
        assert asyncio.run(run()) is None

    assert "continuing without VTube Studio" in caplog.text


def test_errors_inside_an_unavailable_session_are_not_chained_to_the_vts_error():
    fake = FakeVTS(connect_error=OSError("connection refused"))

    async def run() -> None:
        async with vts_session(VTSConfig(), client_factory=lambda _config: fake):
            raise RuntimeError("later failure")

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.__context__ is None
