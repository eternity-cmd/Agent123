"""
job_extract.py

岗位文本信息抽取模块。

功能：
1. 从岗位明细 DataFrame 中构造大模型输入；
2. 调用 llm_interface_layer.llm_service.call_llm("job_extract", ...)；
3. 清洗并解析大模型返回结果；
4. 把结构化岗位画像合并回原始 DataFrame；
5. 保存为新的 CSV 文件。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from llm_interface_layer.llm_service import call_llm


DEFAULT_JOB_PROFILE: Dict[str, Any] = {
    "standard_job_name": "",
    "job_category": "",
    "degree_requirement": "",
    "major_requirement": "",
    "experience_requirement": "",
    "hard_skills": [],
    "tools_or_tech_stack": [],
    "certificate_requirement": [],
    "soft_skills": [],
    "practice_requirement": "",
    "job_level": "",
    "suitable_student_profile": "",
    "raw_requirement_summary": "",
}

LIST_FIELDS = {
    "hard_skills",
    "tools_or_tech_stack",
    "certificate_requirement",
    "soft_skills",
}


def clean_text(value: object) -> str:
    """基础文本清洗。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_first_existing_value(record: pd.Series, candidates: Sequence[str]) -> str:
    """从多个字段候选里取第一个非空值。"""
    for field in candidates:
        if field in record.index:
            text = clean_text(record.get(field, ""))
            if text:
                return text
    return ""


def parse_json_like_text(value: object) -> Dict[str, Any]:
    """
    兼容解析：
    - dict
    - 纯 JSON 字符串
    - ```json 代码块
    - 前后带少量解释文本
    """
    if isinstance(value, dict):
        return value

    text = clean_text(value)
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def normalize_list_value(value: object) -> List[str]:
    """
    把列表字段统一转成字符串列表。
    支持：
    - list
    - JSON 数组字符串
    - 逗号/顿号/分号分隔字符串
    """
    if value is None:
        return []

    if isinstance(value, list):
        return sorted({clean_text(item) for item in value if clean_text(item)})

    text = clean_text(value)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return sorted({clean_text(item) for item in parsed if clean_text(item)})
        except json.JSONDecodeError:
            pass

    parts = re.split(r"[、,，;/；|]+", text)
    return sorted({clean_text(part) for part in parts if clean_text(part)})


def normalize_profile_fields(
    parsed_result: Dict[str, Any],
    record: pd.Series,
) -> Dict[str, Any]:
    """
    把大模型返回结果标准化到固定 schema。

    兼容两类情况：
    1. 模型直接按目标 schema 返回；
    2. 当前 mock 或旧接口返回其他字段名，需要映射到目标 schema。
    """
    profile = dict(DEFAULT_JOB_PROFILE)

    # 先填当前记录里的标准岗位名
    profile["standard_job_name"] = get_first_existing_value(
        record,
        ["standard_job_name", "normalized_job_title", "job_title_norm", "job_name", "job_title"],
    )

    # 直接字段映射
    for field in DEFAULT_JOB_PROFILE:
        if field in parsed_result and field != "standard_job_name":
            profile[field] = parsed_result[field]

    # 兼容旧字段名
    profile["job_category"] = clean_text(
        profile["job_category"]
        or parsed_result.get("job_family", "")
        or parsed_result.get("job_category", "")
    )
    profile["degree_requirement"] = clean_text(
        profile["degree_requirement"]
        or parsed_result.get("education_requirement", "")
        or parsed_result.get("degree_requirement", "")
    )
    profile["major_requirement"] = clean_text(
        profile["major_requirement"] or parsed_result.get("major_requirement", "")
    )
    profile["experience_requirement"] = clean_text(
        profile["experience_requirement"]
        or parsed_result.get("experience_requirement", "")
    )
    profile["hard_skills"] = normalize_list_value(
        profile["hard_skills"] or parsed_result.get("skills", [])
    )
    profile["tools_or_tech_stack"] = normalize_list_value(
        profile["tools_or_tech_stack"] or parsed_result.get("tools_or_tech_stack", [])
    )
    if not profile["tools_or_tech_stack"]:
        profile["tools_or_tech_stack"] = list(profile["hard_skills"])

    profile["certificate_requirement"] = normalize_list_value(
        profile["certificate_requirement"] or parsed_result.get("certificates", [])
    )
    profile["soft_skills"] = normalize_list_value(
        profile["soft_skills"] or parsed_result.get("soft_skills", [])
    )
    profile["practice_requirement"] = clean_text(
        profile["practice_requirement"] or parsed_result.get("practice_requirement", "")
    )
    profile["job_level"] = clean_text(
        profile["job_level"] or parsed_result.get("job_level", "")
    )
    profile["suitable_student_profile"] = clean_text(
        profile["suitable_student_profile"] or parsed_result.get("suitable_student_profile", "")
    )
    profile["raw_requirement_summary"] = clean_text(
        profile["raw_requirement_summary"]
        or parsed_result.get("summary", "")
        or parsed_result.get("raw_requirement_summary", "")
    )

    # 再兜底标准岗位名
    profile["standard_job_name"] = clean_text(
        parsed_result.get("standard_job_name", "") or profile["standard_job_name"]
    )

    return profile


def validate_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """保证所有字段存在且类型稳定。"""
    fixed = dict(DEFAULT_JOB_PROFILE)
    for field, default_value in DEFAULT_JOB_PROFILE.items():
        if field in LIST_FIELDS:
            fixed[field] = normalize_list_value(profile.get(field, default_value))
        else:
            fixed[field] = clean_text(profile.get(field, default_value))

    return fixed


def build_extraction_input(record: pd.Series) -> Dict[str, Any]:
    """构造发送给 LLM 的岗位输入。"""
    return {
        "job_name": get_first_existing_value(record, ["job_name", "job_title", "job_title_norm"]),
        "standard_job_name": get_first_existing_value(
            record,
            ["standard_job_name", "normalized_job_title", "job_title_norm", "job_name", "job_title"],
        ),
        "industry": get_first_existing_value(record, ["industry"]),
        "company_name": get_first_existing_value(record, ["company_name", "company_name_norm"]),
        "company_type": get_first_existing_value(record, ["company_type", "company_type_norm"]),
        "company_size": get_first_existing_value(record, ["company_size", "company_size_norm"]),
        "city": get_first_existing_value(record, ["city", "job_address_norm", "job_address"]),
        "salary_raw": get_first_existing_value(
            record,
            ["salary_raw", "salary_range_raw", "salary_range", "salary_range_clean"],
        ),
        "job_desc": get_first_existing_value(
            record,
            ["job_desc", "job_description_clean", "job_description_text", "job_description"],
        ),
        "company_desc": get_first_existing_value(
            record,
            ["company_desc", "company_description_clean", "company_description_text", "company_description"],
        ),
    }


def call_job_extract_llm(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """调用统一 LLM 接口。"""
    extra_context = {
        "project": "基于AI的大学生职业规划智能体",
        "output_language": "zh-CN",
        "output_schema": DEFAULT_JOB_PROFILE,
        "instruction": (
            "请严格按照 output_schema 返回 JSON。"
            "如果某项信息无法确定，请返回空字符串或空列表。"
        ),
    }
    return call_llm(
        "job_extract",
        input_data=input_data,
        context_data=None,
        student_state=None,
        extra_context=extra_context,
    )


def extract_job_profile(record: pd.Series) -> Dict[str, Any]:
    """
    单条岗位抽取函数。

    返回固定结构，并附带：
    - extract_success
    - extract_error
    - job_extract_json
    """
    input_data = build_extraction_input(record)

    try:
        llm_response = call_job_extract_llm(input_data)
        parsed = parse_json_like_text(llm_response)
        normalized = normalize_profile_fields(parsed, record)
        validated = validate_profile(normalized)
        validated["extract_success"] = True
        validated["extract_error"] = ""
        validated["job_extract_json"] = json.dumps(validated, ensure_ascii=False)
        return validated
    except Exception as exc:
        fallback = validate_profile({"standard_job_name": input_data.get("standard_job_name", "")})
        fallback["extract_success"] = False
        fallback["extract_error"] = clean_text(str(exc))
        fallback["job_extract_json"] = json.dumps(fallback, ensure_ascii=False)
        return fallback


def convert_profile_to_row(profile: Dict[str, Any]) -> Dict[str, Any]:
    """把抽取结果转换成适合 DataFrame/CSV 的行结构。"""
    row = {}
    for field in DEFAULT_JOB_PROFILE:
        if field in LIST_FIELDS:
            row[field] = json.dumps(profile.get(field, []), ensure_ascii=False)
        else:
            row[field] = clean_text(profile.get(field, ""))
    row["extract_success"] = bool(profile.get("extract_success", False))
    row["extract_error"] = clean_text(profile.get("extract_error", ""))
    row["job_extract_json"] = clean_text(profile.get("job_extract_json", ""))
    return row


def merge_extraction_results(
    df: pd.DataFrame,
    extracted_df: pd.DataFrame,
    index_col: str = "_source_index",
) -> pd.DataFrame:
    """将抽取结果合并回原始 DataFrame。"""
    return df.merge(extracted_df, on=index_col, how="left")


def batch_extract_job_profiles(
    df: pd.DataFrame,
    log_every: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    批处理岗位抽取。

    返回：
    - merged_df: 原始数据 + 抽取字段
    - extracted_df: 仅抽取结果
    """
    working_df = df.copy().reset_index(drop=False).rename(columns={"index": "_source_index"})
    cache: Dict[str, Dict[str, Any]] = {}
    extracted_rows: List[Dict[str, Any]] = []

    total = len(working_df)
    print(f"[job_extract] Start extracting job profiles, total rows: {total}")

    for i, (_, row) in enumerate(working_df.iterrows(), start=1):
        input_data = build_extraction_input(row)
        cache_key = "||".join(
            [
                input_data.get("standard_job_name", ""),
                input_data.get("job_desc", "")[:1500],
                input_data.get("company_desc", "")[:800],
            ]
        )

        if cache_key in cache:
            profile = cache[cache_key]
        else:
            profile = extract_job_profile(row)
            cache[cache_key] = profile

        extracted_row = {"_source_index": row["_source_index"], **convert_profile_to_row(profile)}
        extracted_rows.append(extracted_row)

        if i == 1 or i % log_every == 0 or i == total:
            print(f"[job_extract] Progress: {i}/{total}")

    extracted_df = pd.DataFrame(extracted_rows)
    merged_df = merge_extraction_results(working_df, extracted_df, index_col="_source_index")
    return merged_df, extracted_df


def save_extraction_result(merged_df: pd.DataFrame, output_csv_path: str) -> None:
    """保存抽取后的完整结果。"""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def load_table(input_path: str) -> pd.DataFrame:
    """支持从 CSV / Excel 加载输入数据。"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def process_job_extract(
    df: pd.DataFrame,
    output_csv_path: Optional[str] = None,
    log_every: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    主流程函数。

    返回：
    - merged_df: 原始数据 + 抽取字段
    - extracted_df: 抽取结果表
    """
    merged_df, extracted_df = batch_extract_job_profiles(df=df, log_every=log_every)
    if output_csv_path:
        save_extraction_result(merged_df, output_csv_path)
    return merged_df, extracted_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="岗位文本信息抽取")
    parser.add_argument(
        "--input",
        default="outputs/jobs_dedup_result.csv",
        help="输入岗位数据文件路径，支持 CSV / Excel",
    )
    parser.add_argument(
        "--output",
        default="outputs/jobs_extracted.csv",
        help="输出抽取结果 CSV 文件路径",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="每处理多少条打印一次日志",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_table(args.input)
    merged_df, extracted_df = process_job_extract(
        df=df,
        output_csv_path=args.output,
        log_every=args.log_every,
    )

    success_count = int(merged_df["extract_success"].fillna(False).sum()) if "extract_success" in merged_df.columns else 0
    print("[job_extract] Finished.")
    print(f"[job_extract] Input rows: {len(df)}")
    print(f"[job_extract] Extracted rows: {len(extracted_df)}")
    print(f"[job_extract] Success rows: {success_count}")
    print(f"[job_extract] Output file: {args.output}")


if __name__ == "__main__":
    main()


