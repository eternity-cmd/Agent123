"""
llm_client.py

统一的大模型底层调用客户端。
当前默认提供 mock 调用，预留真实 OpenAI / ChatGPT API 接入位置。
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict
from urllib import request

from .config import DEFAULT_LLM_CONFIG, LLMConfig
from .job_pipeline_mock import mock_job_dedup_result, mock_job_extract_result
from .schemas import TaskType


class LLMClient:
    """底层模型调用客户端。输入 prompt，输出原始文本。"""

    def __init__(self, config: LLMConfig = DEFAULT_LLM_CONFIG) -> None:
        self.config = config

    def generate(
        self,
        task_type: "TaskType | str",
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """统一生成入口，返回模型原始文本。"""
        normalized_task = TaskType.normalize(task_type)

        if self.config.mock_enabled:
            return self._mock_generate(normalized_task, system_prompt, user_prompt)

        last_error: Exception | None = None
        for _ in range(self.config.retry_times + 1):
            try:
                return self._real_generate(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)

        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

    def _real_generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        真实模型调用预留位。
        当前只保留 OpenAI 兼容接口示例，不写真实 API Key。
        """
        api_key = os.getenv(self.config.api_key_env_name, "").strip()
        if not self.config.api_base_url or not api_key:
            raise RuntimeError(
                "Real LLM call requires api_base_url and API key env var, "
                f"env name: {self.config.api_key_env_name}"
            )

        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        req = request.Request(
            self.config.api_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            response_json = json.loads(resp.read().decode("utf-8"))

        return response_json["choices"][0]["message"]["content"]

    def _mock_generate(
        self,
        task_type: TaskType,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """mock 模式：返回可解析的 JSON 字符串。"""
        context_payload = self._extract_context_payload(user_prompt)
        input_data = context_payload.get("input_data", {}) if isinstance(context_payload, dict) else {}

        if task_type == TaskType.JOB_EXTRACT:
            result = mock_job_extract_result(input_data)
        elif task_type == TaskType.JOB_DEDUP:
            result = mock_job_dedup_result(input_data)
        elif task_type == TaskType.RESUME_PARSE:
            result = {
                "name": "张三",
                "gender": "男",
                "phone": "13800000000",
                "email": "student@example.com",
                "school": "某某大学",
                "major": "计算机科学与技术",
                "degree": "本科",
                "graduation_year": "2026",
                "skills": ["Python", "SQL", "机器学习"],
                "certificates": ["CET-6"],
                "project_experience": [
                    {"project_name": "课程推荐系统", "role": "开发", "description": "负责数据处理与模型实验"}
                ],
                "internship_experience": [
                    {"company_name": "某科技公司", "position": "数据分析实习生", "description": "参与报表分析"}
                ],
                "raw_summary": "mock 简历解析结果",
            }
        elif task_type == TaskType.JOB_PROFILE:
            target_job = input_data.get("target_job_name", "") or "数据分析师"
            result = {
                "standard_job_name": target_job,
                "job_category": "数据类",
                "required_degree": "本科及以上",
                "preferred_majors": ["计算机科学与技术", "统计学", "数据科学"],
                "required_skills": ["Python", "SQL", "数据分析", "可视化"],
                "required_certificates": [],
                "soft_skills": ["沟通能力", "学习能力"],
                "vertical_paths": [f"{target_job} -> 高级{target_job}", f"高级{target_job} -> 数据分析负责人"],
                "transfer_paths": [f"{target_job} -> 数据产品经理", f"{target_job} -> BI工程师"],
                "job_summary": f"{target_job} 需要具备数据处理、分析表达和业务理解能力。",
            }
        elif task_type == TaskType.STUDENT_PROFILE:
            result = {
                "skill_profile": {"Python": "熟悉", "SQL": "熟悉", "机器学习": "入门"},
                "certificate_profile": ["CET-6"],
                "soft_skill_profile": {"沟通能力": "良好", "学习能力": "较强"},
                "complete_score": 82.0,
                "competitiveness_score": 76.0,
                "strengths": ["有 Python/SQL 基础", "有项目经历"],
                "weaknesses": ["业务分析经验偏少", "缺少正式实习深度沉淀"],
                "summary": "mock 学生画像结果",
            }
        elif task_type == TaskType.JOB_MATCH:
            result = {
                "overall_score": 78.0,
                "basic_requirement_score": 85.0,
                "skill_score": 80.0,
                "professional_quality_score": 76.0,
                "growth_potential_score": 82.0,
                "strengths": ["学历满足要求", "Python/SQL 技能匹配度较高"],
                "gaps": ["行业项目经验不足", "缺少 BI 工具实战证明"],
                "improvement_suggestions": ["补充 1 个真实业务分析项目", "学习 PowerBI/Tableau 并产出作品集"],
                "summary": "mock 人岗匹配结果",
            }
        elif task_type == TaskType.CAREER_PATH_PLAN:
            result = {
                "primary_target_job": "数据分析师",
                "backup_target_jobs": ["BI分析师", "数据运营"],
                "direct_path": ["数据分析实习生", "数据分析师", "高级数据分析师"],
                "transition_path": ["数据运营", "BI分析师", "数据分析师"],
                "short_term_plan": ["3个月内补齐 SQL/BI 项目作品集", "完善一版面向数据岗位的简历"],
                "mid_term_plan": ["6-12个月争取数据分析实习", "沉淀行业分析方法论"],
                "risk_notes": ["避免只学工具不做项目", "定期根据招聘要求调整技能栈"],
                "summary": "mock 职业路径规划结果",
            }
        elif task_type == TaskType.CAREER_REPORT:
            report_text = (
                "职业规划报告\n"
                "目标岗位：数据分析师\n"
                "总体结论：当前能力基础较好，但需要补强业务项目经验和 BI 工具实战。\n"
                "建议路径：数据分析实习生 -> 数据分析师 -> 高级数据分析师。\n"
            )
            result = {
                "report_title": "大学生职业规划报告",
                "target_job": "数据分析师",
                "match_summary": "岗位基础要求匹配较好，技能项存在可补齐差距。",
                "path_summary": "优先走数据分析实习生到数据分析师的直接路径，备选 BI 分析方向。",
                "action_summary": "短期补项目与 BI 工具，中期争取实习并持续迭代简历。",
                "report_text": report_text,
                "report_sections": {
                    "career_goal": "数据分析师",
                    "core_gap": ["BI 工具实战", "业务分析项目经验"],
                },
            }
        else:
            result = {"message": "unsupported mock task"}

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _extract_context_payload(user_prompt: str) -> Dict[str, Any]:
        """从 user_prompt 中提取拼装后的上下文 dict，供 mock 模式使用。"""
        text = user_prompt or ""
        marker = "输入上下文 JSON："
        if marker in text:
            tail = text.split(marker, 1)[-1].strip()
            try:
                return json.loads(tail)
            except json.JSONDecodeError:
                pass

        matches = re.findall(r"\{[\s\S]*\}", text)
        if not matches:
            return {}
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            return {}


