"""
长期记忆：跨会话记住用户的事实与偏好。

存储：一张 user_memories 表（user_id + key + value），结构化，可直接查/改/删。
写入：每轮回答结束后，用一个轻量 LLM 调用抽取「值得长期记住」的信息（后台执行，不阻塞回答）。
读取：每轮开始前取出该用户的记忆，渲染成一段文本注入 system prompt。

和短期记忆的分工：
- 短期记忆：会话维度的多轮上下文  → LangGraph checkpointer（已有）
- 长期记忆：用户维度的事实与偏好  → user_memories 表（本次新增）
"""
import json
import logging
import re

import app.config as config
from app.config import get_llm

logger = logging.getLogger(__name__)

# 这些是「没有真实身份」的共享兜底值。写进去等于所有人共用一份记忆，必须排除。
_NO_MEMORY_USERS = {"", "anonymous", "default"}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_memories (
    user_id    TEXT        NOT NULL,
    mem_key    TEXT        NOT NULL,
    mem_value  TEXT        NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, mem_key)
)
"""

EXTRACT_PROMPT = """从下面这轮对话里抽取「值得长期记住」的用户信息。

只抽取：
- 用户稳定的身份/属性（姓名、部门、岗位、所在城市）
- 长期偏好（希望回答简短、喜欢表格、不要表情符号）
- 长期目标或正在做的项目

不要抽取：
- 一次性的问题内容（例如「今天问了年假几天」）
- 临时状态、情绪、闲聊
- 密码、证件号、手机号等敏感信息

没有可抽取的内容就返回空数组。只输出 JSON，不要任何解释：
{{"memories": [{{"key": "英文蛇形标识，如 name / department / answer_style", "value": "中文内容"}}]}}

【用户说】
{user_message}

【助手回答】
{assistant_reply}"""

# 模板自检：str.format 会把未转义的 { 当成占位符，
# 这里在 import 阶段就验证一次，写错了直接启动失败，而不是等到线上第一次抽取时才静默报错。
EXTRACT_PROMPT.format(user_message="", assistant_reply="")


async def ensure_memory_table(pool) -> None:
    """幂等建表，在 lifespan 里跟 checkpointer.setup() 一起调。"""
    async with pool.connection() as conn:
        await conn.execute(CREATE_TABLE_SQL)


async def load_memories(pool, user_id: str, limit: int | None = None) -> list[dict]:
    """取出该用户的记忆，最近更新的优先。"""
    if user_id in _NO_MEMORY_USERS:
        return []
    limit = limit or config.MEMORY_MAX_ITEMS
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT mem_key, mem_value
            FROM user_memories
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        return list(await cur.fetchall())


def render_memories(memories: list[dict]) -> str:
    """把记忆渲染成注入 system prompt 的一段文本；没有记忆时返回空串。"""
    if not memories:
        return ""
    lines = "\n".join(f"- {m['mem_key']}: {m['mem_value']}" for m in memories)
    return (
        "【关于当前用户的长期记忆】\n"
        f"{lines}\n"
        "使用要求：只在和当前问题相关时自然使用，不要逐条复述，不要生硬地说「根据我的记忆」。"
    )


async def clear_memories(pool, user_id: str) -> int:
    """清空某个用户的记忆（给「忘记我」这类功能用）。"""
    async with pool.connection() as conn:
        cur = await conn.execute("DELETE FROM user_memories WHERE user_id = %s", (user_id,))
        return cur.rowcount


async def extract_and_save(pool, user_id: str, user_message: str, assistant_reply: str) -> None:
    """
    抽取并落库。

    这是旁路逻辑：任何异常都必须吞掉——记忆写失败不能影响用户已经拿到的回答，
    更不能因为一次抽取超时把整个请求带崩。
    """
    if not config.MEMORY_ENABLED:
        return
    if user_id in _NO_MEMORY_USERS:
        # 没有真实身份就不写，否则所有匿名用户共享一份记忆
        return
    if not user_message or not assistant_reply:
        return

    try:
        prompt = EXTRACT_PROMPT.format(
            user_message=user_message[:2000],
            assistant_reply=assistant_reply[:2000],
        )
        response = await get_llm().ainvoke(prompt)
        payload = _parse_json(_as_text(response.content))
        memories = payload.get("memories") or []

        rows = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()[:64]
            value = str(item.get("value") or "").strip()[:500]
            if key and value:
                rows.append((user_id, key, value))

        if not rows:
            return

        async with pool.connection() as conn:
            # psycopg3 的 AsyncConnection 只有 execute，批量执行必须走 cursor
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO user_memories (user_id, mem_key, mem_value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, mem_key)
                    DO UPDATE SET mem_value = EXCLUDED.mem_value, updated_at = now()
                    """,
                    rows,
                )
        logger.info("记忆已更新 user_id=%s keys=%s", user_id, [r[1] for r in rows])

    except Exception:
        logger.exception("记忆抽取失败 user_id=%s", user_id)


def _as_text(content) -> str:
    """不同模型返回的 content 可能是 str，也可能是内容块列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _parse_json(text: str) -> dict:
    """
    宽容地解析模型输出。

    即使提示词要求「只输出 JSON」，模型仍可能裹一层 ```json 代码块或加一句解释，
    所以先剥代码块，再退化为「截取第一个 { 到最后一个 }」。
    """
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
