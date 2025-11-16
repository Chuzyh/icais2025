import requests
import json

url = "http://localhost:3000/literature_review"
data = {"query": "Summarize the state-of-the-art in AI-assisted drug discovery and molecular design."}

# 启动 POST 请求
with requests.post(url, json=data, stream=True) as r:
    full_text = ""
    for line in r.iter_lines():
        if line:
            if line.decode() == "data: [DONE]":
                break
            # 每一行都是类似 "data: {json字符串}"
            if line.decode().startswith("data: "):
                content = line.decode().replace("data: ", "")
                try:
                    parsed = json.loads(content)
                    delta = parsed["choices"][0]["delta"].get("content", "")
                    full_text += delta
                    print(delta, end="", flush=True)
                except Exception:
                    pass

print("===== 完整回答 =====")
print(full_text)
