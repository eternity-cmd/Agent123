"""
job_pipeline_mock.py

岗位流水线（job_extract / job_dedup）在 mock 模式下的确定性返回逻辑，
供 LLMClient 在 mock 模式下复用（逻辑与早期独立岗位 LLM 脚本一致）。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_title_rule(title: str) -> str:
    title = re.sub(r"[()（）【】\[\]\s]+", "", title or "")
    title = re.sub(r"(校招|社招|急聘|直招|双休|五险一金|接受小白|可实习转正)", "", title)
    title = title.replace("前端开发", "前端开发工程师")
    title = title.replace("后端开发", "后端开发工程师")
    title = title.replace("Java开发", "Java开发工程师")
    title = title.replace("测试开发", "测试开发工程师")
    return title


def _infer_job_family(title: str) -> str:
    rules = [
        ("前端", "前端开发"),
        ("后端", "后端开发"),
        ("java", "Java开发"),
        ("python", "Python开发"),
        ("测试", "测试"),
        ("数据分析", "数据分析"),
        ("数据", "数据开发"),
        ("算法", "算法"),
        ("实施", "实施交付"),
        ("运维", "运维"),
        ("产品", "产品"),
        ("ui", "UI设计"),
        ("设计", "设计"),
        ("销售", "销售"),
        ("技术支持", "技术支持"),
    ]
    lowered = (title or "").lower()
    for keyword, family in rules:
        if keyword in lowered:
            return family
    return "其他"


def _infer_job_level(title: str) -> str:
    if "实习" in (title or ""):
        return "实习"
    if any(x in (title or "") for x in ["高级", "资深", "专家", "架构师"]):
        return "高级"
    if any(x in (title or "") for x in ["主管", "经理", "负责人"]):
        return "管理"
    return "普通"


def _extract_skills(text: str) -> List[str]:
    skill_keywords = [
        "Python", "Java", "C++", "C", "Go", "JavaScript", "TypeScript", "HTML", "CSS",
        "Vue", "React", "Angular", "Node.js", "Django", "Flask", "FastAPI", "Spring",
        "MySQL", "SQL", "Oracle", "PostgreSQL", "Redis", "MongoDB", "Linux", "Git",
        "Docker", "Kubernetes", "Nginx", "Hadoop", "Spark", "Flink", "TensorFlow",
        "PyTorch", "Pandas", "NumPy", "Excel", "PowerBI", "Tableau", "ArcGIS", "CASS",
        "MES", "ERP", "WMS", "IoT", "NLP",
    ]
    lowered = (text or "").lower()
    found = [skill for skill in skill_keywords if skill.lower() in lowered]
    return sorted(set(found))


def _extract_soft_skills(text: str) -> List[str]:
    mapping = {
        "沟通能力": ["沟通能力", "沟通技巧", "表达能力"],
        "学习能力": ["学习能力", "好学", "主动学习"],
        "抗压能力": ["抗压能力", "抗压", "承压"],
        "团队协作": ["团队协作", "团队合作", "协作精神"],
        "责任心": ["责任心", "认真负责", "细心"],
        "执行力": ["执行力", "推动能力"],
    }
    return [soft for soft, keywords in mapping.items() if any(k in (text or "") for k in keywords)]


def _extract_certificates(text: str) -> List[str]:
    candidates = ["软考", "PMP", "CET-4", "CET-6", "教师资格证", "计算机二级"]
    return [item for item in candidates if item.lower() in (text or "").lower()]


def _extract_education(text: str) -> str:
    for item in ["博士", "硕士", "本科", "大专", "中专"]:
        if item in (text or ""):
            return item
    return ""


def _extract_experience(text: str) -> Dict[str, Any]:
    cleaned = text or ""
    match = re.search(r"(\d+)\s*[-~至]?\s*(\d+)?\s*年.*经验", cleaned)
    if match:
        return {"experience_requirement": match.group(0), "experience_years_min": int(match.group(1))}
    match = re.search(r"(\d+)\s*年(?:以上)?", cleaned)
    if match:
        return {"experience_requirement": match.group(0), "experience_years_min": int(match.group(1))}
    if "应届" in cleaned or "实习" in cleaned:
        return {"experience_requirement": "应届/实习可投", "experience_years_min": 0}
    return {"experience_requirement": "", "experience_years_min": None}


def _extract_responsibilities(text: str) -> List[str]:
    cleaned = re.sub(r"[；;]", "\n", text or "")
    lines = [line.strip(" 1234567890.、:-") for line in cleaned.splitlines() if line.strip()]
    lines = [line for line in lines if len(line) >= 6]
    return lines[:5]


def mock_job_extract_result(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    与早期岗位画像抽取接口返回字段对齐，
    job_extract.normalize_profile_fields 会再映射到标准 schema。
    """
    title = input_data.get("job_name") or input_data.get("standard_job_name", "")
    job_text = input_data.get("job_desc", "")
    company_text = input_data.get("company_desc", "")
    merged_text = "\n".join([str(title), str(job_text), str(company_text)]).strip()
    exp = _extract_experience(merged_text)
    skills = _extract_skills(merged_text)
    soft_skills = _extract_soft_skills(merged_text)
    certificates = _extract_certificates(merged_text)
    responsibilities = _extract_responsibilities(str(job_text))
    keywords = sorted(set([_infer_job_family(str(title)), *skills[:6], *soft_skills[:4]]))
    std = clean_text(input_data.get("standard_job_name", "")) or str(title)
    return {
        "standard_job_name": std,
        "job_family": _infer_job_family(str(title)),
        "job_category": _infer_job_family(str(title)),
        "job_level": _infer_job_level(str(title)),
        "education_requirement": _extract_education(merged_text),
        "degree_requirement": _extract_education(merged_text),
        "experience_requirement": exp["experience_requirement"],
        "experience_years_min": exp["experience_years_min"],
        "skills": skills,
        "hard_skills": skills,
        "certificates": certificates,
        "certificate_requirement": certificates,
        "soft_skills": soft_skills,
        "keywords": keywords,
        "responsibilities": responsibilities,
        "summary": responsibilities[0] if responsibilities else (str(job_text)[:120] if job_text else ""),
        "raw_requirement_summary": responsibilities[0] if responsibilities else (str(job_text)[:120] if job_text else ""),
        "tools_or_tech_stack": list(skills),
        "major_requirement": "",
        "practice_requirement": "",
        "suitable_student_profile": "",
    }


def mock_job_dedup_result(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    成对去重：根据 titles + records 生成 mappings / duplicate_groups，
    并给出 is_same_standard_job 供 parse_llm_judgement 直接使用。
    """
    titles = input_data.get("titles") or []
    records = input_data.get("records") or []

    mappings = []
    for raw_title in titles:
        rt = clean_text(raw_title)
        if not rt:
            continue
        normalized = _normalize_title_rule(rt)
        mappings.append(
            {
                "raw_title": rt,
                "normalized_title": normalized or rt,
                "job_family": _infer_job_family(normalized or rt),
                "confidence": 0.85,
            }
        )

    duplicate_groups: List[Dict[str, Any]] = []
    is_same = False
    standard_job_name = ""
    confidence = 0.0
    merge_reason = ""

    if len(records) >= 2:
        master = records[0]
        duplicate_ids: List[str] = []
        for rec in records[1:]:
            same_url = master.get("job_url") and master.get("job_url") == rec.get("job_url")
            same_code = master.get("job_code") and master.get("job_code") == rec.get("job_code")
            same_company = master.get("company_name") == rec.get("company_name")
            same_title = master.get("normalized_job_title") == rec.get("normalized_job_title")
            desc_ratio = SequenceMatcher(
                None,
                str(master.get("job_description_text") or "")[:600],
                str(rec.get("job_description_text") or "")[:600],
            ).ratio()
            if same_url or same_code or (same_company and same_title and desc_ratio >= 0.92):
                duplicate_ids.append(str(rec.get("record_id", "")))
        if duplicate_ids:
            duplicate_groups.append(
                {
                    "master_record_id": str(master.get("record_id", "")),
                    "duplicate_record_ids": duplicate_ids,
                    "reason": "mock_similarity_dedup",
                }
            )

    if len(mappings) >= 2:
        n0, n1 = mappings[0]["normalized_title"], mappings[1]["normalized_title"]
        is_same = bool(n0 and n1 and n0 == n1)
        if is_same:
            standard_job_name = n0
            confidence = 0.86
            merge_reason = "llm_same_normalized_title"
        else:
            confidence = 0.45
            merge_reason = "llm_different_normalized_title"
    elif duplicate_groups:
        is_same = True
        confidence = 0.92
        merge_reason = "llm_duplicate_group"
        if records:
            standard_job_name = clean_text(records[0].get("normalized_job_title", ""))

    return {
        "is_same_standard_job": is_same,
        "standard_job_name": standard_job_name,
        "confidence": confidence,
        "merge_reason": merge_reason,
        "mappings": mappings,
        "normalized_titles": mappings,
        "duplicate_groups": duplicate_groups,
    }
