# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json

with open(r'cwq.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== 前5条数据结构 ===')
for i, item in enumerate(data[:5]):
    print(f'\n[{i+1}] ID: {item.get("ID", "")}')
    print(f'    question: {item.get("question", "")[:80]}')
    print(f'    answer字段: {repr(item.get("answer", ""))[:100]}')
    print(f'    answers字段: {repr(item.get("answers", ""))[:100]}')
    # 打印所有字段名
    print(f'    所有字段: {list(item.keys())}')


