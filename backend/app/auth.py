"""Optional single-operator HTTP Basic foundation, including browser media requests."""

import base64
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings


class AccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin not in settings.cors_origins:
                return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        token = settings.auth_token
        if token and request.url.path != "/health" and request.method != "OPTIONS":
            supplied = request.headers.get("authorization", "")
            valid = False
            if supplied.startswith("Basic "):
                try:
                    user, password = (
                        base64.b64decode(supplied[6:], validate=True).decode().split(":", 1)
                    )
                    valid = user == "scenemind" and secrets.compare_digest(
                        password.encode(), token.encode()
                    )
                except (ValueError, UnicodeError):
                    pass
            elif supplied.startswith("Bearer "):
                valid = secrets.compare_digest(supplied[7:].encode(), token.encode())
            if not valid:
                return JSONResponse(
                    {"detail": "Authentication required"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="SceneMind"'},
                )
        return await call_next(request)
