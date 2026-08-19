"""多提供商 LLM 接入层：每个 Skill 可独立配置模型与 API key。

支持两类端点：
  - anthropic          官方 Anthropic API
  - openai_compatible  一切 OpenAI 兼容端点（OpenAI/DeepSeek/Qwen/GLM/中转站…）

配置存于 data/skill_models.json（UI 可编辑，明文本地保存，勿提交仓库）：
{
  "text_observation": {
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "api_key": "sk-...",
    "base_url": null
  },
  "d1_direct_scoring": { "provider": "openai_compatible", ... }
}

复现性要求：call() 返回响应中的**实际模型名**并写入审计字段。
中转 API 可能静默换模型，只记请求参数会破坏复现性。
"""
from __future__ import annotations

import json
import os
import time

import config as C

CONFIG_PATH = C.DATA_ROOT / "skill_models.json"

# 推理强度：off=不启用；low/medium/high 映射到各提供商的机制
#   openai_compatible → 请求体 reasoning_effort（gpt-5/o 系列；不支持的端点自动降级）
#   anthropic         → extended thinking 预算（注意：开启后 API 强制 temperature=1，
#                        输出不再逐字确定，观察类 Skill 建议保持 off 以保复现性）
# 档位说明（各 API 的真实上限不同，勿混淆）：
#   OpenAI 官方 API 的 reasoning_effort 合法值只有 minimal/low/medium/high，
#   没有 "max" —— 聊天客户端里的 "max" 是应用层概念，不是 API 参数。
#   Anthropic thinking 是数值预算，无档位名，max 在这里映射为 32k 预算。
#   因此本项目的 "max"：anthropic → 32768 预算；openai → 按 high 发送。
REASONING_LEVELS = ("off", "low", "medium", "high", "max")
ANTHROPIC_THINKING_BUDGET = {"low": 2048, "medium": 8192,
                             "high": 16384, "max": 32768}

# 中转站 WAF 常拦截 SDK 的 x-stainless-* 遥测头与默认 UA（实测 403
# "Your request was blocked"），openai_compatible 一律走裸 httpx + 浏览器 UA。
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")

# 最近一次 call() 的 usage 统计（审计与成本指标用）。
# 特别关注 input tokens 异常膨胀：说明中转在注入自己的系统提示词，
# 这是不受 prompt_hash 追踪的隐藏变量，正式实验前必须核查。
LAST_USAGE: dict = {}

DEFAULTS = {
    "text_observation": {
        "provider": "anthropic", "model": "claude-sonnet-5",
        "api_key": "", "base_url": None, "reasoning": "off",
    },
    "d1_direct_scoring": {
        "provider": "anthropic", "model": "claude-sonnet-5",
        "api_key": "", "base_url": None, "reasoning": "off",
    },
}


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for skill, item in saved.items():
                cfg.setdefault(skill, {}).update(item or {})
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def resolve(skill: str, cli_model: str | None = None) -> dict:
    """CLI --model 覆盖 > 配置文件 > 默认。api_key 缺省回退环境变量。"""
    cfg = load_config().get(skill) or dict(DEFAULTS.get(skill) or
                                           DEFAULTS["text_observation"])
    cfg = dict(cfg)
    if cli_model:
        cfg["model"] = cli_model
    if not cfg.get("api_key"):
        env = ("ANTHROPIC_API_KEY" if cfg.get("provider") == "anthropic"
               else "OPENAI_API_KEY")
        cfg["api_key"] = os.environ.get(env, "")
        if not cfg["api_key"] and cfg.get("provider") == "anthropic":
            f = C.DATA_ROOT / "anthropic_key.txt"
            if f.exists():
                cfg["api_key"] = f.read_text(encoding="utf-8").strip()
    return cfg


def normalize_openai_base(url: str | None) -> str:
    """中转站惯例：base_url 需以 /v1 结尾（SDK/请求在其后拼 /chat/completions）。"""
    u = (url or "https://api.openai.com").rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


def _openai_call(cfg: dict, system_prompt: str, user_message: str,
                 max_tokens: int) -> tuple[str, str]:
    """裸 httpx 实现 OpenAI 兼容调用。

    自适应参数（各上游口味不一，按报错逐项降级并重试）：
      max_completion_tokens ↔ max_tokens、temperature、reasoning_effort
    对中转常见的瞬时 "Upstream request failed" 做 3 次重试。
    """
    import httpx
    url = normalize_openai_base(cfg.get("base_url")) + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "User-Agent": BROWSER_UA, "Content-Type": "application/json"}
    body: dict = {"model": cfg["model"],
                  "max_completion_tokens": max_tokens,
                  "temperature": 0,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_message}]}
    level = cfg.get("reasoning", "off")
    if level != "off":
        # OpenAI API 无 "max" 档，向下映射为其上限 high
        body["reasoning_effort"] = "high" if level == "max" else level

    # 流式必需，不是优化：中转的前置 CDN 会掐断"长时间无字节"的连接，
    # 在阻塞式调用下表现为 SSL UNEXPECTED_EOF——推理越久越必然触发。
    # 开 stream 后服务端立刻回 SSE 分片，连接不再空闲。
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    global LAST_USAGE
    adaptations, transient = 0, 0
    while True:
        err = ""
        try:
            pieces: list[str] = []
            model_seen, usage = "", {}
            with httpx.stream("POST", url, json=body, headers=headers,
                              timeout=300) as r:
                if r.status_code != 200:
                    err = r.read().decode("utf-8", "replace")[:500]
                else:
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            j = json.loads(chunk)
                        except ValueError:
                            continue
                        model_seen = j.get("model") or model_seen
                        usage = j.get("usage") or usage
                        for ch in j.get("choices") or []:
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                pieces.append(piece)
                    LAST_USAGE = usage
                    return "".join(pieces), model_seen or cfg["model"]
        except httpx.TransportError as exc:
            # 连接被中转掐断（RemoteProtocolError / ReadTimeout 等）在拿到响应前
            # 就抛出，走不到下面的状态码分支，必须单独退避重试。
            if transient >= 4:
                raise RuntimeError(f"传输层反复失败: {type(exc).__name__}: {exc}")
            transient += 1
            time.sleep(3 * transient)
            continue

        # 少数上游不认 stream_options，先摘掉它再谈其他参数
        if r.status_code == 400 and "stream_options" in err.lower() \
                and "stream_options" in body:
            body.pop("stream_options")
            continue
        # 参数自适应（最多 3 项）
        if r.status_code == 400 and adaptations < 3:
            low = err.lower()
            if "max_completion_tokens" in low and "max_completion_tokens" in body:
                body["max_tokens"] = body.pop("max_completion_tokens")
                adaptations += 1
                continue
            if "max_tokens" in low and "max_tokens" in body and \
                    "max_completion_tokens" not in low:
                body["max_completion_tokens"] = body.pop("max_tokens")
                adaptations += 1
                continue
            if "temperature" in low and "temperature" in body:
                body.pop("temperature")
                adaptations += 1
                continue
            if "reasoning" in low and "reasoning_effort" in body:
                body.pop("reasoning_effort")
                adaptations += 1
                continue
        # 中转上游抖动：重试
        if ("upstream" in err.lower() or r.status_code in (500, 502, 503, 504)) \
                and transient < 3:
            transient += 1
            time.sleep(2 * transient)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {err[:200]}")


def _anthropic_call(cfg: dict, system_prompt: str, user_message: str,
                    max_tokens: int) -> tuple[str, str]:
    import anthropic
    kw = {"api_key": cfg["api_key"]}
    if cfg.get("base_url"):
        kw["base_url"] = cfg["base_url"]
    client = anthropic.Anthropic(**kw)

    req: dict = {"model": cfg["model"], "max_tokens": max_tokens,
                 "system": system_prompt,
                 "messages": [{"role": "user", "content": user_message}]}
    level = cfg.get("reasoning", "off")
    if level != "off":
        budget = ANTHROPIC_THINKING_BUDGET[level]
        req["thinking"] = {"type": "enabled", "budget_tokens": budget}
        req["max_tokens"] = max(max_tokens, budget + 2000)
        # thinking 开启时 API 禁止 temperature=0（强制 1），不传即可
    else:
        req["temperature"] = 0

    resp = client.messages.create(**req)
    global LAST_USAGE
    u = getattr(resp, "usage", None)
    LAST_USAGE = ({"input_tokens": getattr(u, "input_tokens", None),
                   "output_tokens": getattr(u, "output_tokens", None)}
                  if u else {})
    # thinking 开启时 content 含 thinking 块，取第一个 text 块
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"),
                "")
    return text, getattr(resp, "model", cfg["model"])


def call(cfg: dict, system_prompt: str, user_message: str,
         max_tokens: int = 4000) -> tuple[str, str]:
    """统一调用。返回 (响应文本, 响应中报告的实际模型名)。"""
    provider = cfg.get("provider", "anthropic")
    if provider == "anthropic":
        return _anthropic_call(cfg, system_prompt, user_message, max_tokens)
    if provider == "openai_compatible":
        return _openai_call(cfg, system_prompt, user_message, max_tokens)
    raise ValueError(f"未知 provider: {provider}")


def list_models(cfg: dict) -> list[str]:
    """拉取端点可用模型列表（选型与排错用）。"""
    import httpx
    if cfg.get("provider") == "openai_compatible":
        base = normalize_openai_base(cfg.get("base_url"))
        r = httpx.get(base + "/models",
                      headers={"Authorization": f"Bearer {cfg['api_key']}",
                               "User-Agent": BROWSER_UA}, timeout=30)
    else:
        base = (cfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
        r = httpx.get(base + "/v1/models",
                      headers={"x-api-key": cfg["api_key"],
                               "anthropic-version": "2023-06-01",
                               "User-Agent": BROWSER_UA}, timeout=30)
    r.raise_for_status()
    return sorted(m.get("id", "?") for m in r.json().get("data", []))


PROBE_QUESTION = ("How many prime numbers are there between 100 and 200? "
                  "Reply with only the number.")


def probe_reasoning(cfg: dict) -> tuple[bool | None, str]:
    """实测推理强度是否真实生效（中转站常接受参数却无声丢弃）。

    判据：openai 格式对比 off 与最高档的 completion_tokens 与
    reasoning_tokens；anthropic 检查响应是否含 thinking 块。
    返回 (是否生效, 证据说明)。生效性未知（如网络失败）返回 (None, 原因)。
    """
    try:
        if cfg.get("provider") == "openai_compatible":
            import httpx
            url = normalize_openai_base(cfg.get("base_url")) + "/chat/completions"
            headers = {"Authorization": f"Bearer {cfg['api_key']}",
                       "User-Agent": BROWSER_UA}

            def once(effort):
                body = {"model": cfg["model"], "max_completion_tokens": 20000,
                        "messages": [{"role": "user",
                                      "content": PROBE_QUESTION}]}
                if effort:
                    body["reasoning_effort"] = effort
                r = httpx.post(url, json=body, headers=headers, timeout=300)
                r.raise_for_status()
                u = r.json().get("usage", {}) or {}
                det = u.get("completion_tokens_details") or {}
                return u.get("completion_tokens"), det.get("reasoning_tokens")

            ct0, rt0 = once(None)
            ct1, rt1 = once("high")
            if rt1 and rt1 > 0:
                return True, (f"生效：high 档 reasoning_tokens={rt1}"
                              f"（off 档 {rt0 or 0}）")
            if ct0 and ct1 and ct1 > max(ct0 * 3, ct0 + 100):
                return True, (f"疑似生效：completion_tokens off={ct0} → "
                              f"high={ct1}（无 reasoning_tokens 字段，按用量推断）")
            return False, (f"无效：off/high 的 completion_tokens 相同"
                           f"（{ct0}/{ct1}），无 reasoning_tokens —— "
                           "该端点接受参数但将其丢弃，调档不会改变模型行为")

        # anthropic：thinking 生效时响应必含 thinking 块（API 契约保证）
        import anthropic
        kw = {"api_key": cfg["api_key"]}
        if cfg.get("base_url"):
            kw["base_url"] = cfg["base_url"]
        client = anthropic.Anthropic(**kw)
        resp = client.messages.create(
            model=cfg["model"], max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": PROBE_QUESTION}])
        has_thinking = any(getattr(b, "type", "") == "thinking"
                           for b in resp.content)
        return (True, "生效：响应含 thinking 块") if has_thinking else \
               (False, "无效：请求未报错但响应无 thinking 块")
    except Exception as e:
        return None, f"无法判定（{type(e).__name__}: {str(e)[:140]}）"


def ping(cfg: dict) -> tuple[bool, str]:
    """连通性测试。max_tokens 给足——推理型模型会把小预算全花在思考上。"""
    try:
        _, actual = call(cfg, "Reply with the single word: ok",
                         "ping", max_tokens=512)
        return True, actual
    except Exception as e:
        msg = str(e)
        hints = []
        if "blocked" in msg.lower():
            hints.append("端点 WAF 拦截（已用浏览器头仍被拦：确认 base_url 正确）")
        if "401" in msg or "invalid" in msg.lower() and "key" in msg.lower():
            hints.append("API key 无效或额度耗尽")
        if "404" in msg:
            hints.append("路径不对：base_url 是否需要 /v1，或模型名不存在")
        if "upstream" in msg.lower():
            hints.append("中转上游通道故障：换个模型名试试（用「拉取模型列表」看可用项）")
        tail = ("；".join(hints)) if hints else ""
        return False, f"{type(e).__name__}: {msg[:180]}" + (f"  ← {tail}" if tail else "")


def model_slug(cfg: dict) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9.-]+", "-", cfg.get("model", "unknown"))[:40]
