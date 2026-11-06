from langchain_core.tools import tool

@tool
def get_current_time() -> str:
    """
    获取当前时间。当用户询问时间时调用。

    Returns:
        str: 当前时间，格式为 "YYYY-MM-DD HH:MM:SS"
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")