# -*- coding: utf-8 -*-
"""
v4 数据准备脚本（最终版）：
  - 职业资格：正则去「非贪婪+排除.」，补上中文书名号/冒号风格 fallback；等级清洗 ISBN/编号
  - 版次：版次正则 *? 非贪婪 + 支持中文数字（第十版）
  - 分类编号：14374724(屈光手术学)→ophthalmology，15227083(儿童视觉发育诊断与治疗)→optometry
"""
import re
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(r"f:/StudyMaterials/medical-rag/抽取结果")
DST_ROOT = Path(r"f:/StudyMaterials/medical-rag/data")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CN_NUM = r"[一二三四五六七八九十百千零两\d]+"
GRADE_NOISE_RE = re.compile(r"(978\d{10}|(?<![A-Za-z])\d{7,10}(?![A-Za-z])|第[一二三四五六七八九十百千零两\d]+\s*版)")

# ========== 分类规则（按顺序匹配，第一个命中为准） ==========
RULES = [
    (
        lambda f: any(k in f.name for k in [
            "WILLS", "威尔斯", "Wills",
            "眼科学（第十版）", "眼科学基础", "眼科学 第3版", "眼科学 第",
            "青光眼", "葡萄膜炎", "视网膜", "神经眼科", "眼眶病", "小儿眼科",
            "低视力学", "斜视弱视学", "屈光手术",
            "14374724",  # 屈光手术学（纯编号源文件）
        ]),
        DST_ROOT / "diseases" / "ophthalmology"
    ),
    (
        lambda f: any(k in f.name for k in [
            "斜视与弱视", "双眼视觉学", "临床双眼视觉学",
            "接触镜", "硬性角膜接触镜", "接触镜学",
            "眼镜学",
            "眼视光器械学", "眼视光公共卫生学",
            "视光师手册", "门诊视光师",
            "视觉训练的原理",
            "儿童视觉发育",
            "15227083",  # 儿童视觉发育诊断与治疗（纯编号源文件）
        ]),
        DST_ROOT / "diseases" / "optometry"
    ),
    (
        lambda f: any(k in f.name for k in [
            "眼镜定配工", "眼镜验光员",
            "门诊笔记",
            "儿童视光",
            "视觉与学习",
        ]),
        DST_ROOT / "health_articles"
    ),
]

# ========== 清洗正则 ==========
PREFIX_RE = re.compile(r"^datalab-output-?")
PDF_RE = re.compile(r"\.pdf$")
ISBN_RE = re.compile(r"978\d{10}")
NUM8D_RE = re.compile(r"(?<![A-Za-z0-9])_?\d{7,10}_?")
DOTS_RE = re.compile(r"\.{3,}")
CN_BOOKNAME_HINT_RE = re.compile(
    r"(眼科学|青光眼|葡萄膜炎|视网膜|神经眼科|眼眶病|小儿眼科|低视力学|斜视弱视学|"
    r"斜视与弱视|双眼视觉学|临床双眼视觉学|硬性角膜接触镜|接触镜学|眼镜学|"
    r"眼视光器械学|眼视光公共卫生学|视光师手册|视觉训练的原理|视觉与学习|"
    r"儿童视光|屈光手术学|儿童视觉发育|眼科学基础|眼科学|"
    r"眼镜定配工|眼镜验光员|门诊笔记)"
)
BRACKET_FIX_RE = re.compile(rf"（[^）]*?第\s*{CN_NUM}\s*版$")
PUBLISH_TAIL_RE = re.compile(
    r"[_\s]*作[_ ]*者[：:][^\n]*?出版发行[_ ]*[_ ]*北京[:：][^.]*人民卫生出版社[^.]*"
    r"|[_ ]{2,}20\d{2}[\.年]\d{0,2}[\.月]?\d{0,2}$"
)
LEFT_BRACKET_ONLY_RE = re.compile(r"[（(][^）)\s]*$")
VER_RE = re.compile(rf"[（(][^）)]*?第\s*{CN_NUM}\s*版[）)]|第\s*{CN_NUM}\s*版")

# 职业资格第一优先：扫描件 「眼镜定配工（初级）」括号风格
VOC_PAT1 = re.compile(r"(眼镜定配工|眼镜验光员)[_\s]*[（(]([^）)]+)")
# 职业资格第二优先：书籍电子档 「眼镜验光员：高级」「《眼镜验光员 技师 高级技师》」冒号/书名号风格
VOC_PAT2 = re.compile(r"(眼镜定配工|眼镜验光员)[_\s：:]*[《]?([^》）)\.；;]+)")

CIP_RE = re.compile(
    r"^\s*([^/]+?)\s*/\s*[^/]*?(著|译|编|主译|主编)\b"
)

H1_BAD_WORDS = {"目录", "目 录", "引言", "序", "绪论", "名家述评", "寄语", "前言"}


def extract_title_from_cip_or_text(md_path: Path) -> str | None:
    """三级兜底：1) 文件名中文暗示命中直接用 2) CIP行/ 3) 有效H1"""
    name = md_path.name
    m = CN_BOOKNAME_HINT_RE.search(name)
    if m:
        return m.group(1)  # 第一优先级：直接从文件名抓
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # 2) 找 CIP 风格的「书名 / 作者 著」行
    BAD_TITLE_SET = {
        "图书在版编目(CIP)数据", "图书在版编目 (CIP) 数据",
        "图书在版编目数据", "CIP数据", "在版编目数据",
        "目 录", "目录",
    }
    for line in text.splitlines()[:40]:
        s = line.strip()
        mc = CIP_RE.match(s)
        if mc:
            title = mc.group(1).strip()
            title = re.sub(r"[（(](图书|CIP|在版编目|数据[^）)]*)[）)]", "", title)
            title = title.strip(" ：:·.()（）")
            if (
                2 <= len(title) <= 40
                and title not in BAD_TITLE_SET
                and "CIP" not in title
                and "在版编目" not in title
            ):
                return title
    # 3) 找有效 H1（排除序言目录类）
    for line in text.splitlines()[:50]:
        s = line.strip()
        if s.startswith("#"):
            t = s.lstrip("#").strip()
            if (
                2 <= len(t) <= 30
                and re.search(r"[\u4e00-\u9fa5]", t)
                and t not in H1_BAD_WORDS
                and not any(b in t for b in H1_BAD_WORDS)
            ):
                return t
    return None


def clean_name(md_path: Path) -> str:
    name = md_path.name
    stem = md_path.stem  # 去 .md
    stem = PREFIX_RE.sub("", stem)
    stem = PDF_RE.sub("", stem)
    stem = ISBN_RE.sub("", stem)

    # 职业资格类：先试括号风格，再试冒号/书名号风格
    grade = None
    m_job = VOC_PAT1.search(name)
    if not m_job:
        m_job = VOC_PAT2.search(name)
    if m_job:
        job, g = m_job.group(1), m_job.group(2).strip()
        g = GRADE_NOISE_RE.sub("", g)
        g = g.rstrip("）)》")
        g = re.sub(r"[\.\s_]+$", "", g).strip()
        title_seed = f"{job}（{g}）"
        stem = title_seed
    else:
        # 其他：若文件名里已经直接能看到书名关键词，优先用（不读文件）
        m = CN_BOOKNAME_HINT_RE.search(stem)
        if m:
            title_seed = m.group(1)
            ver_match = VER_RE.search(stem)
            suffix_ver = ""
            if ver_match:
                suffix_ver = ver_match.group(0)
                if suffix_ver.startswith("（") and "）" not in suffix_ver:
                    suffix_ver = suffix_ver + "）"
                if suffix_ver.startswith("(") and ")" not in suffix_ver:
                    suffix_ver = suffix_ver + ")"
            stem = title_seed + (suffix_ver or "")
        else:
            fallback = extract_title_from_cip_or_text(md_path)
            if fallback:
                stem = fallback

    # 通用清杂
    stem = NUM8D_RE.sub("", stem)
    stem = DOTS_RE.sub("", stem)
    stem = PUBLISH_TAIL_RE.sub("", stem)
    stem = stem.replace("+", "_")

    # 先做两端与空白清理（注意：这里不能 strip 括号类字符！否则会把末尾「）」剥掉）
    stem = stem.strip(" _-·.,:：—–[]【】、")
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"_+", "_", stem)

    # 之后再补括号闭合：版次缺右括号「（第十版」→「（第十版）」
    bm = BRACKET_FIX_RE.search(stem)
    if bm:
        stem = stem[: bm.end()] + "）" + stem[bm.end():]
    # 其他左括号未闭合：「眼镜验光员（初级」→补右括号
    m2 = LEFT_BRACKET_ONLY_RE.search(stem)
    if m2:
        close = "）" if "（" in m2.group(0) else ")"
        stem = stem + close

    if not stem:
        stem = md_path.stem
    return stem + ".md"


def match_category(md_path: Path) -> Path:
    for rule_fn, dst in RULES:
        if rule_fn(md_path):
            return dst
    return DST_ROOT / "health_articles"


def clear_data_dirs():
    for d in [
        DST_ROOT / "diseases" / "ophthalmology",
        DST_ROOT / "diseases" / "optometry",
        DST_ROOT / "health_articles",
        DST_ROOT / "guidelines",
    ]:
        if d.exists():
            for f in d.glob("*.md"):
                f.unlink()


def main():
    clear_data_dirs()
    md_files = list(SRC_ROOT.rglob("*.md"))
    print(f"找到 {len(md_files)} 个源 md 文件")

    stats = {}
    for md in sorted(md_files):
        dst_dir = match_category(md)
        new_name = clean_name(md)
        dst = dst_dir / new_name
        i = 1
        while dst.exists():
            dst = dst_dir / f"{Path(new_name).stem}_{i}.md"
            i += 1
        try:
            shutil.copy2(md, dst)
            k = str(dst.parent.relative_to(DST_ROOT)).replace("\\", "/")
            stats[k] = stats.get(k, 0) + 1
            rel_src = str(md.relative_to(SRC_ROOT)).replace("\\", "/")
            print(f"OK  {rel_src:55s}  ->  {k}/{dst.name}")
        except Exception as e:
            print(f"FAIL {md}: {e}")

    print("\n===== 汇总 =====")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    total = sum(stats.values())
    print(f"共 {total} / 原 {len(md_files)} 个文件")


if __name__ == "__main__":
    main()
