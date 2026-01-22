import requests
from flask import Flask, request, Response
import json
import os
import time

# ================= 配置区域 =================
TARGET_URL = 'https://open.bigmodel.cn/api/anthropic'
LOG_FILE = 'trace_2.txt'

# True  = 打印所有 details
# False = 只打印最终合并后的内容
SHOW_RAW_LLM_STREAM = False
# ===========================================

app = Flask(__name__)


def safe_json_format(text):
    """格式化 JSON"""
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except:
        return text


def get_req_model_name(req_text):
    """从请求体提取模型名称"""
    try:
        data = json.loads(req_text)
        return data.get('model', 'Unknown')
    except:
        return 'Unknown'


def get_resp_model_name(resp_text):
    """从响应流中提取实际执行的模型名称"""
    lines = resp_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("data: "):
            json_str = line[6:]
            if json_str == "[DONE]": continue
            try:
                data = json.loads(json_str)

                # 1. OpenAI / vLLM 标准格式 (通常直接在 data 根节点)
                if 'model' in data:
                    return data['model']

                # 2. Anthropic 格式 (在 message_start 事件中)
                if data.get('type') == 'message_start':
                    return data.get('message', {}).get('model', 'Unknown')

            except:
                pass
    return "Unknown"


def extract_stream_summary(text):
    """解析 SSE 流式响应内容"""
    lines = text.split('\n')
    full_content = []
    content_type = None
    is_stream = False

    for line in lines:
        line = line.strip()
        if line.startswith("data: "):
            is_stream = True
            json_str = line[6:]
            if json_str == "[DONE]":
                continue
            try:
                data = json.loads(json_str)
                # Anthropic 格式
                if data.get('type') == 'content_block_delta':
                    delta = data.get('delta', {})
                    if not content_type and 'type' in delta:
                        content_type = delta['type']
                    if 'text' in delta:
                        full_content.append(delta['text'])
                # OpenAI 格式
                elif 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        full_content.append(delta['content'])
            except:
                pass

    if not is_stream or not full_content:
        return None

    merged_text = "".join(full_content)
    return (
        f"\n{'=' * 10} Stream Summary {'=' * 10}\n"
        f"Content Type: {content_type if content_type else 'text'}\n"
        f"Full Content: {merged_text}\n"
        f"{'=' * 36}\n"
    )


def write_log(req_body, resp_body):
    """格式化日志并输出"""
    req_str = req_body.decode('utf-8', errors='ignore')

    # 1. 提取双方模型名称
    req_model = get_req_model_name(req_str)
    resp_model = get_resp_model_name(resp_body)

    formatted_req = safe_json_format(req_str)
    stream_summary = extract_stream_summary(resp_body)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 2. 构造日志结构
    log_parts = [
        f"\n{'=' * 20} {timestamp} {'=' * 20}",

        # --- Request 部分 ---
        f"🔵 Claude code (Request Model: {req_model})",
        f"{formatted_req}\n",

        # --- Response 部分 ---
        f"🟢 LLM (Response Model: {resp_model})"
    ]

    if SHOW_RAW_LLM_STREAM or stream_summary is None:
        formatted_resp = safe_json_format(resp_body)
        log_parts.append("(Raw Details)")
        log_parts.append(f"{formatted_resp}")
    else:
        log_parts.append("(Raw Details Hidden)")

    if stream_summary:
        log_parts.append(stream_summary)

    log_parts.append(f"{'=' * 50}\n")

    log_content = "\n".join(log_parts)

    print(log_content)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_content)
    except Exception as e:
        print(f"❌ Failed to write to file: {e}")


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{TARGET_URL}/{path}"
    req_body = request.get_data()
    headers = {key: value for (key, value) in request.headers if key != 'Host'}

    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=req_body,
            cookies=request.cookies,
            allow_redirects=False,
            stream=True
        )

        def generate():
            full_response_chunks = []
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    full_response_chunks.append(chunk)
                    yield chunk

            full_resp_text = b"".join(full_response_chunks).decode('utf-8', errors='ignore')
            write_log(req_body, full_resp_text)

        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]

        return Response(generate(), resp.status_code, headers)

    except Exception as e:
        print(f"❌ Error: {e}")
        return str(e), 500


if __name__ == '__main__':
    print(f"🕵️  Sniffer running on port 9000 -> forwarding to {TARGET_URL}")
    print(f"⚙️  Show Raw Stream: {SHOW_RAW_LLM_STREAM}")
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True)