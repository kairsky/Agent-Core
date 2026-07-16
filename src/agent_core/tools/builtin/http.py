"""HTTP GET tool with a hard cap on response body size."""

from __future__ import annotations

from typing import Any

import httpx

from agent_core.tools.base import ToolContext, ToolResult, ToolSpec

MAX_BODY_BYTES = 100_000


class HttpGetTool:
    spec = ToolSpec(
        name="http_get",
        description=(
            "Fetch a URL via HTTP GET. Returns the response body as text, "
            f"truncated to {MAX_BODY_BYTES} bytes."
        ),
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL to fetch"}},
            "required": ["url"],
        },
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = arguments["url"]
        if not url.startswith(("http://", "https://")):
            return ToolResult.failure("validation_error", "URL must start with http:// or https://")
        try:
            async with (
                httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client,
                client.stream("GET", url) as response,
            ):
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) >= MAX_BODY_BYTES:
                        break
                text = body[:MAX_BODY_BYTES].decode("utf-8", errors="replace")
                return ToolResult(
                    ok=response.is_success,
                    content=text if response.is_success else f"HTTP {response.status_code}",
                    data={"status_code": response.status_code, "bytes": len(body)},
                    error_type=None if response.is_success else "http_error",
                )
        except httpx.TimeoutException:
            return ToolResult.failure("timeout", f"Request to {url} timed out.")
        except httpx.HTTPError as exc:
            return ToolResult.failure("execution_error", f"Request failed: {type(exc).__name__}")
