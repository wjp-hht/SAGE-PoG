# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终检查：确保没有遗漏任何军事问题
"""

import json
from pathlib import Path
from collections import Counter

# 读取数据
cwq_path = Path(r"ComplexWebQuestions_test.json")
v2_path = Path(r"src\experiment_results\military_questions_v2_20260504_193832.json")

with open(cwq_path, 'r', encoding='utf-8') as f:
    all_data = json.load(f)

with open(v2_path, 'r', encoding='utf-8') as f:
    v2_data = json.load(f)

extracted_ids = set(item['ID'] for item in v2_data['questions'])

print("=" * 80)
print("最终遗漏检查")
print("=" * 80)
print(f"总问题数: {len(all_data)}")
print(f"已提取: {len(extracted_ids)}")
print(f"未提取: {len(all_data) - len(extracted_ids)}")

# 扩展的军事关键词列表
ALL_MILITARY_KEYWORDS = [
    # 战争名称
    'world war', 'wwi', 'ww1', 'wwii', 'ww2',
    'vietnam war', 'korean war', 'gulf war', 'iraq war', 'afghanistan war',
    'civil war', 'revolutionary war', 'cold war', 'napoleonic war',
    
    # 战役
    'battle', 'siege', 'campaign', 'offensive', 'invasion',
    
    # 军事人员
    'soldier', 'troop', 'veteran', 'combatant',
    'general', 'admiral', 'colonel', 'major', 'captain', 'lieutenant', 'sergeant',
    'commander', 'officer',
    
    # 军事组织
    'army', 'navy', 'marine', 'air force',
    'regiment', 'battalion', 'brigade', 'division', 'corps',
    'squadron', 'fleet', 'platoon', 'infantry', 'cavalry', 'artillery',
    
    # 军事装备
    'weapon', 'arms', 'ammunition',
    'tank', 'aircraft', 'bomber', 'fighter', 'warship', 'submarine',
    'missile', 'nuclear', 'artillery',
    
    # 军事概念
    'military', 'combat', 'conflict', 'warfare',
    'defense', 'attack', 'strategic', 'tactical',
    'peacekeeping', 'deployment',
    
    # 军事奖章
    'medal of honor', 'purple heart', 'bronze star', 'silver star',
    
    # 其他
    'fort', 'base', 'camp'
]

# 检查未提取的问题
print("\n检查未提取的问题中是否包含军事关键词...")
print("-" * 80)

missed_matches = {}
for item in all_data:
    if item['ID'] in extracted_ids:
        continue
    
    question = item.get('question', '').lower()
    matched_keywords = [kw for kw in ALL_MILITARY_KEYWORDS if kw in question]
    
    if matched_keywords:
        missed_matches[item['ID']] = {
            'item': item,
            'keywords': matched_keywords,
            'count': len(matched_keywords)
        }

print(f"发现 {len(missed_matches)} 个未提取但包含军事关键词的问题")

# 按匹配数排序
sorted_missed = sorted(missed_matches.items(), key=lambda x: x[1]['count'], reverse=True)

print("\n前50个可能遗漏的问题（按匹配关键词数排序）:")
print("-" * 80)

truly_military = []
for i, (qid, data) in enumerate(sorted_missed[:50], 1):
    item = data['item']
    keywords = data['keywords']
    count = data['count']
    question = item.get('question', '')
    ctype = item.get('compositionality_type', 'unknown')
    
    print(f"\n{i}. [匹配{count}个词] [{ctype}]")
    print(f"   问题: {question}")
    print(f"   关键词: {', '.join(keywords[:5])}")
    
    # 判断是否真的是军事问题
    q_lower = question.lower()
    is_truly_military = False
    reason = []
    
    # 强军事特征
    if any(war in q_lower for war in ['world war', 'vietnam war', 'korean war', 'civil war', 'gulf war', 'iraq war']):
        is_truly_military = True
        reason.append("包含具体战争名称")
    
    if 'battle of' in q_lower or 'battle for' in q_lower:
        is_truly_military = True
        reason.append("包含具体战役")
    
    if 'military commander' in q_lower or 'military conflict' in q_lower:
        is_truly_military = True
        reason.append("包含军事指挥官/冲突")
    
    if any(unit in q_lower for unit in ['regiment', 'battalion', 'brigade', 'infantry']):
        is_truly_military = True
        reason.append("包含军事单位")
    
    if 'invasion of' in q_lower or 'siege of' in q_lower:
        is_truly_military = True
        reason.append("包含军事行动")
    
    # 排除误匹配
    exclude_patterns = [
        'major league', 'world series', 'baseball',
        'award nominated', 'based on',
        'general motors', 'general electric',
        'captain america', 'seven nation army'
    ]
    if any(pattern in q_lower for pattern in exclude_patterns):
        is_truly_military = False
        reason = ["误匹配（体育/电影/公司名）"]
    
    if is_truly_military:
        print(f"   >>> 真正的军事问题！原因: {'; '.join(reason)}")
        truly_military.append(item)

print("\n" + "=" * 80)
print(f"发现 {len(truly_military)} 个真正遗漏的军事问题！")
print("=" * 80)

if truly_military:
    print("\n遗漏的军事问题列表:")
    for i, item in enumerate(truly_military, 1):
        print(f"{i}. [{item.get('compositionality_type', 'unknown')}] {item.get('question', '')}")
    
    # 保存遗漏的问题
    output_path = Path(r"src\experiment_results\truly_missed_military.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "count": len(truly_military),
            "questions": truly_military
        }, f, ensure_ascii=False, indent=2)
    print(f"\n遗漏的问题已保存到: {output_path}")
else:
    print("\n没有发现真正遗漏的军事问题！")

# 统计最常见的遗漏关键词
print("\n" + "=" * 80)
print("统计分析:")
print("=" * 80)

all_missed_keywords = []
for data in missed_matches.values():
    all_missed_keywords.extend(data['keywords'])

keyword_freq = Counter(all_missed_keywords)
print("\n最常见的遗漏关键词（Top 20）:")
for kw, freq in keyword_freq.most_common(20):
    print(f"  {kw}: {freq}次")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
if truly_military:
    print(f"发现 {len(truly_military)} 个真正遗漏的军事问题")
    print("建议: 将这些问题加入数据集")
else:
    print("当前数据集已经很完整，没有明显遗漏")
    print("v2版本（81个问题）可以作为最终版本使用")
