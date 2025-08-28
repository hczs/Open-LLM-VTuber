import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    """
    提取 OpenAI 返回中包含的 JSON 字符串（可能被 ```json 包裹），并转成 dict
    """
    # 正则匹配 ```json ... ``` 或 ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 如果没有 ```json 包裹，就假设整个 text 就是 JSON
        json_str = text.strip()

    return json.loads(json_str)
