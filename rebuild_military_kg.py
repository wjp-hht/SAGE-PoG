# -*- coding: utf-8 -*-
"""
重建 military_kg_with_names.nt

从 3 个 military_kg_group*.nt（纯关系三元组）和 freebase-rdf-latest.gz（全量 FreeBase dump）
中提取实体名称，合并生成带名称的完整 KG 文件。

输出: military_kg_with_names.nt

用法:
    python rebuild_military_kg.py

预计耗时: 1-4 小时（取决于磁盘速度，32GB gz 流式解压为主瓶颈）
内存占用: ~200-500 MB（仅存实体 ID 集合）
"""

import os
import sys
import gzip
import time
import re

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUP_FILES = [
    os.path.join(BASE_DIR, "military_kg_group1.nt"),
    os.path.join(BASE_DIR, "military_kg_group2.nt"),
    os.path.join(BASE_DIR, "military_kg_group3.nt"),
]

FREEBASE_GZ = os.path.join(BASE_DIR, "freebase-rdf-latest.gz")
OUTPUT_FILE = os.path.join(BASE_DIR, "military_kg_with_names.nt")

# FreeBase 实体/属性全 URI 前缀
NS = "http://rdf.freebase.com/ns/"
RDF_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
TYPE_OBJECT_NAME = NS + "type.object.name"

# ============================================================
# Step 1: 收集所有唯一实体 ID
# ============================================================
def collect_entity_ids(group_files):
    """扫描所有 group 文件，收集出现过的实体 URI（subject / object）。"""
    entity_ids = set()
    # URI 匹配模式: <http://rdf.freebase.com/ns/m.xxxx> 或 g.xxxx
    uri_re = re.compile(r'<http://rdf\.freebase\.com/ns/([mg]\.\S+?)>')
    total_lines = 0

    for fpath in group_files:
        if not os.path.exists(fpath):
            print(f"[WARN] 跳过不存在的文件: {fpath}")
            continue
        print(f"[Step 1] 扫描: {os.path.basename(fpath)} ...")
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                for m in uri_re.finditer(line):
                    entity_ids.add(m.group(1))  # 存 "m.02p717m" 或 "g.121sr23g"
                if total_lines % 100000 == 0:
                    print(f"         已扫描 {total_lines} 行, 唯一实体 {len(entity_ids):,}", end="\r")
    print(f"\n[Step 1] 完成: 共 {total_lines:,} 行三元组, {len(entity_ids):,} 个唯一实体 ID")
    return entity_ids


# ============================================================
# Step 2: 从 FreeBase dump 提取名称三元组
# ============================================================
def extract_names_from_dump(entity_ids, gz_path, output_nametmp):
    """流式读取 gz dump，匹配 target 实体集合中 type.object.name / rdfs:label 三元组。

    写入临时名称文件 output_nametmp，同时返回写入条数。
    """
    if not os.path.exists(gz_path):
        print(f"[FATAL] FreeBase dump 不存在: {gz_path}")
        return 0

    # 构建快速查找：URI 片段 -> 是否在集合中
    # dump 里 subject URI 格式: <http://rdf.freebase.com/ns/m.02p717m> (Tab分隔)
    # 我们用更快的子串查找: 取 <...ns/ 之后、> 之前的部分
    print(f"[Step 2] 开始流式解压 {os.path.basename(gz_path)} ...")
    print(f"         目标实体数: {len(entity_ids):,}")
    print(f"         预计耗时较长，请耐心等待...")

    written = 0
    scanned_lines = 0
    matched_lines = 0
    t_start = time.time()

    # 预计算一些用于快速判断的字符串
    NAME_STR = "type.object.name"
    LABEL_STR = "rdf-schema#label"

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f_in, \
         open(output_nametmp, "w", encoding="utf-8") as f_out:

        for line in f_in:
            scanned_lines += 1

            # 快速跳过：必须包含 type.object.name 或 rdfs:label 才可能是名称行
            if NAME_STR not in line and LABEL_STR not in line:
                if scanned_lines % 5000000 == 0:
                    _report_progress(scanned_lines, matched_lines, written, t_start)
                continue

            # 解析 subject
            # 格式: <http://rdf.freebase.com/ns/ENTITY_ID>\t<PREDICATE>\t<VALUE>
            try:
                close_bracket = line.index(">", 0)
            except ValueError:
                continue
            subject_uri = line[:close_bracket + 1]  # e.g. <http://rdf.freebase.com/ns/m.02p717m>

            # 提取 entity id
            if not subject_uri.startswith("<" + NS):
                continue
            entity_id = subject_uri[len("<" + NS):-1]  # m.02p717m

            if entity_id not in entity_ids:
                continue

            matched_lines += 1

            # 将行格式从 Tab 分隔转为空格分隔（对齐 NT 标准格式），并写入
            # 原始: <s>\t<p>\t<o>\t.
            # 目标: <s> <p> <o> .
            nt_line = _convert_to_nt_format(line)
            if nt_line:
                f_out.write(nt_line)
                written += 1

            if scanned_lines % 5000000 == 0:
                _report_progress(scanned_lines, matched_lines, written, t_start)

    elapsed = time.time() - t_start
    print(f"\n[Step 2] 完成: 扫描 {scanned_lines:,} 行, 命中 {matched_lines:,} 条, "
          f"写入 {written:,} 条名称三元组, 耗时 {elapsed/60:.1f} 分钟")
    return written


def _convert_to_nt_format(tab_line):
    """将 Tab 分隔的 dump 行转为标准空格分隔 NT 格式"""
    ls = tab_line.strip()
    if not ls:
        return None
    # 按 Tab 分割
    parts = ls.split("\t")
    if len(parts) < 3:
        return None
    # parts 大概: [<s>, <p>, <o>, .]  或 [<s>, <p>, "literal"@en, .]
    # 用空格重新连接
    return " ".join(p.strip() for p in parts if p.strip()) + "\n"


def _report_progress(scanned, matched, written, t_start):
    elapsed = time.time() - t_start
    rate = scanned / elapsed if elapsed > 0 else 0
    print(f"         已扫描 {scanned:,} 行 | 命中 {matched:,} | 写入 {written:,} | "
          f"{rate:,.0f} 行/秒", end="\r")


# ============================================================
# Step 3: 合并生成最终文件
# ============================================================
def merge_all(group_files, name_file, output_path):
    """合并关系三元组 + 名称三元组 → 最终 KG 文件"""
    t_start = time.time()
    total = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        # 先写关系三元组
        for fpath in group_files:
            if not os.path.exists(fpath):
                continue
            print(f"[Step 3] 合并关系: {os.path.basename(fpath)} ...")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    ls = line.strip()
                    if ls:
                        f_out.write(ls + "\n")
                        total += 1

        # 再写名称三元组
        if os.path.exists(name_file):
            print(f"[Step 3] 合并名称: {os.path.basename(name_file)} ...")
            with open(name_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    ls = line.strip()
                    if ls:
                        f_out.write(ls + "\n")
                        total += 1

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Step 3] 完成: {total:,} 条三元组, {size_mb:.1f} MB, 耗时 {elapsed:.1f} 秒")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("重建 military_kg_with_names.nt")
    print("=" * 65)
    print(f"工作目录 : {BASE_DIR}")
    print(f"Group 文件: {len(GROUP_FILES)} 个")
    print(f"FreeBase  : {os.path.basename(FREEBASE_GZ)}")
    print(f"输出文件  : {os.path.basename(OUTPUT_FILE)}")
    print("=" * 65)

    # 检查输入
    missing_groups = [f for f in GROUP_FILES if not os.path.exists(f)]
    if missing_groups:
        print(f"[FATAL] 缺少 group 文件: {missing_groups}")
        sys.exit(1)
    if not os.path.exists(FREEBASE_GZ):
        print(f"[FATAL] 缺少 FreeBase dump: {FREEBASE_GZ}")
        sys.exit(1)

    name_tmp = os.path.join(BASE_DIR, "_military_names_tmp.nt")

    try:
        # Step 1: 收集实体 ID
        entity_ids = collect_entity_ids(GROUP_FILES)
        if not entity_ids:
            print("[FATAL] 未找到任何实体 ID")
            sys.exit(1)

        # Step 2: 从 dump 提取名称
        name_count = extract_names_from_dump(entity_ids, FREEBASE_GZ, name_tmp)
        if name_count == 0:
            print("[WARN] 未提取到任何名称三元组！最终文件仅含关系三元组。")
            # 仍继续，至少合并关系部分

        # Step 3: 合并
        merge_all(GROUP_FILES, name_tmp, OUTPUT_FILE)

    finally:
        # 清理临时文件
        if os.path.exists(name_tmp):
            os.remove(name_tmp)
            print("[清理] 已删除临时名称文件")

    # 验证
    final_size = os.path.getsize(OUTPUT_FILE)
    print("\n" + "=" * 65)
    print(f"[DONE] 输出: {OUTPUT_FILE}")
    print(f"       大小: {final_size / (1024 * 1024):.1f} MB")
    print(f"       路径: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
