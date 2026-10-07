import json
import mimetypes
import os
import secrets
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class TikTokIntegrationError(RuntimeError):
    pass


class TikTokIntegration:
    AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
    TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
    API_BASE = "https://open.tiktokapis.com"

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / ".tiktok"
        self.token_file = self.data_dir / "tokens.json"
        self.state_file = self.data_dir / "oauth_state.json"
        self.data_dir.mkdir(exist_ok=True)
        try:
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass

    @staticmethod
    def quote(value: str) -> str:
        return quote(value, safe="")

    def _config(self):
        client_key = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
        client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()
        redirect_uri = os.getenv(
            "TIKTOK_REDIRECT_URI",
            "https://cortes.imobiprohub.com/auth/tiktok/callback",
        ).strip()
        if not client_key or not client_secret:
            raise TikTokIntegrationError(
                "TikTok ainda não foi configurado no servidor. "
                "Defina TIKTOK_CLIENT_KEY e TIKTOK_CLIENT_SECRET no .env."
            )
        return client_key, client_secret, redirect_uri

    def _write_private_json(self, path: Path, data: dict):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        temp.replace(path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _request(self, url: str, method="GET", headers=None, data=None, json_body=None):
        request_headers = dict(headers or {})
        body = data
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json; charset=UTF-8")
        elif isinstance(data, dict):
            body = urlencode(data).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        req = Request(url, data=body, headers=request_headers, method=method)
        try:
            with urlopen(req, timeout=60) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type or raw.startswith(b"{"):
                    return json.loads(raw.decode("utf-8"))
                return raw
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
                error = parsed.get("error") or parsed
                if isinstance(error, dict):
                    detail = error.get("message") or error.get("code") or detail
            except json.JSONDecodeError:
                pass
            raise TikTokIntegrationError(f"TikTok retornou erro HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TikTokIntegrationError(f"Não foi possível conectar ao TikTok: {exc.reason}") from exc

    def authorization_url(self) -> str:
        client_key, _, redirect_uri = self._config()
        state = secrets.token_urlsafe(32)
        self._write_private_json(self.state_file, {"state": state, "created_at": time.time()})
        params = {
            "client_key": client_key,
            "response_type": "code",
            "scope": "user.info.basic,video.publish,video.upload",
            "redirect_uri": redirect_uri,
            "state": state,
        }
        return self.AUTH_URL + "?" + urlencode(params)

    def exchange_code(self, code: str, state: str):
        expected = self._read_json(self.state_file)
        if (
            not expected
            or not secrets.compare_digest(str(expected.get("state", "")), state)
            or time.time() - float(expected.get("created_at", 0)) > 600
        ):
            raise TikTokIntegrationError("Estado OAuth inválido ou expirado. Inicie a conexão novamente.")
        client_key, client_secret, redirect_uri = self._config()
        token = self._request(
            self.TOKEN_URL,
            method="POST",
            data={
                "client_key": client_key,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )
        if not token.get("access_token"):
            raise TikTokIntegrationError("O TikTok não retornou um access_token.")
        now = time.time()
        token["obtained_at"] = now
        token["expires_at"] = now + int(token.get("expires_in", 86400))
        token["refresh_expires_at"] = now + int(token.get("refresh_expires_in", 31536000))
        self._write_private_json(self.token_file, token)
        self.state_file.unlink(missing_ok=True)
        return token

    def _refresh(self, token: dict) -> dict:
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise TikTokIntegrationError("Sessão do TikTok expirada. Conecte a conta novamente.")
        client_key, client_secret, _ = self._config()
        refreshed = self._request(
            self.TOKEN_URL,
            method="POST",
            data={
                "client_key": client_key,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
        )
        if not refreshed.get("access_token"):
            raise TikTokIntegrationError("Não foi possível renovar a sessão do TikTok.")
        now = time.time()
        refreshed["obtained_at"] = now
        refreshed["expires_at"] = now + int(refreshed.get("expires_in", 86400))
        refreshed["refresh_expires_at"] = now + int(refreshed.get("refresh_expires_in", 31536000))
        self._write_private_json(self.token_file, refreshed)
        return refreshed

    def _token(self) -> dict:
        token = self._read_json(self.token_file)
        if not token.get("access_token"):
            raise TikTokIntegrationError("Nenhuma conta TikTok está conectada.")
        if time.time() >= float(token.get("expires_at", 0)) - 300:
            token = self._refresh(token)
        return token

    def status(self):
        configured = bool(os.getenv("TIKTOK_CLIENT_KEY")) and bool(os.getenv("TIKTOK_CLIENT_SECRET"))
        token = self._read_json(self.token_file)
        return {
            "configured": configured,
            "connected": bool(token.get("access_token")),
            "scopes": token.get("scope", ""),
            "open_id": token.get("open_id"),
        }

    def disconnect(self):
        self.token_file.unlink(missing_ok=True)
        self.state_file.unlink(missing_ok=True)

    def _api_json(self, path: str, body: dict):
        token = self._token()
        response = self._request(
            self.API_BASE + path,
            method="POST",
            headers={"Authorization": "Bearer " + token["access_token"]},
            json_body=body,
        )
        error = response.get("error") or {}
        if error and error.get("code") not in (None, "", "ok"):
            raise TikTokIntegrationError(error.get("message") or error.get("code"))
        return response.get("data") or {}

    def creator_info(self):
        return self._api_json("/v2/post/publish/creator_info/query/", {})

    def publish_video(
        self,
        video_path: Path,
        *,
        title: str,
        privacy_level: str,
        disable_comment: bool,
        disable_duet: bool,
        disable_stitch: bool,
        is_aigc: bool,
    ):
        creator = self.creator_info()
        allowed_privacy = creator.get("privacy_level_options") or []
        if privacy_level not in allowed_privacy:
            raise TikTokIntegrationError("Privacidade inválida para esta conta TikTok.")

        video_path = Path(video_path)
        total_size = video_path.stat().st_size
        max_chunk = 64 * 1024 * 1024
        preferred_chunk = 10 * 1024 * 1024
        if total_size <= max_chunk:
            chunk_size = total_size
            total_chunks = 1
        else:
            chunk_size = preferred_chunk
            total_chunks = (total_size + chunk_size - 1) // chunk_size

        init_data = self._api_json(
            "/v2/post/publish/video/init/",
            {
                "post_info": {
                    "title": title,
                    "privacy_level": privacy_level,
                    "disable_duet": bool(disable_duet),
                    "disable_comment": bool(disable_comment),
                    "disable_stitch": bool(disable_stitch),
                    "is_aigc": bool(is_aigc),
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": total_size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": total_chunks,
                },
            },
        )
        upload_url = init_data.get("upload_url")
        publish_id = init_data.get("publish_id")
        if not upload_url or not publish_id:
            raise TikTokIntegrationError("TikTok não retornou URL de upload.")

        mime = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
        if mime not in {"video/mp4", "video/quicktime", "video/webm"}:
            mime = "video/mp4"

        with video_path.open("rb") as fh:
            start = 0
            while start < total_size:
                chunk = fh.read(chunk_size)
                end = start + len(chunk) - 1
                self._request(
                    upload_url,
                    method="PUT",
                    headers={
                        "Content-Type": mime,
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {start}-{end}/{total_size}",
                    },
                    data=chunk,
                )
                start = end + 1

        return {"publish_id": publish_id, "status": "PROCESSING_UPLOAD"}

    def publish_status(self, publish_id: str):
        return self._api_json("/v2/post/publish/status/fetch/", {"publish_id": publish_id})
