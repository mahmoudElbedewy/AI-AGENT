"""
يحمّل الدوال المساعدة (pure functions) من consumers.py بدون الحاجة لتثبيت
Django/Channels/LangChain/إلخ.

طريقة العمل: بنقرأ الملف بـ AST ونستخرج بس تعريفات الدوال والمتغيرات اللي
محتاجينها للاختبار (دوال التصنيف/regex)، وننفذها (exec) في namespace فيه
المكتبات القليلة المطلوبة فعلاً (re, datetime).

ده مفيد لأن نفس الملف بيستورد مكتبات تقيلة (langchain, langgraph, docx,
openpyxl, ...) غير متاحة في بيئة الاختبار، وهي مش مطلوبة لاختبار منطق
الـ regex/التصنيف نفسه.
"""

import ast
import re
from datetime import datetime

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_HERE, "consumers.py")

CONSUMERS_PATH = os.environ.get("CONSUMERS_PATH", _DEFAULT_PATH)

# الدوال والمتغيرات اللي عايزين نستخرجها للاختبار
NAMES_TO_EXTRACT = {
    "_INTERNAL_LEAK_PATTERNS",
    "_contains_internal_leak",
    "_normalize_arabic_query",
    "_is_pdf_related_question",
    "_needs_code_execution",
    "_needs_file_creation",
    "_needs_image_generation",
    "_is_image_generation_followup",
    "_needs_live_search",
    "_is_summary_request",
    "_wants_table",
    "_is_arabic_text",
    "_has_actionable_upload_request",
    "_llm_text",
    "_query_terms",
}


def load_helpers(path=CONSUMERS_PATH):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    extracted_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES_TO_EXTRACT:
            extracted_nodes.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in NAMES_TO_EXTRACT:
                    extracted_nodes.append(node)

    module = ast.Module(body=extracted_nodes, type_ignores=[])
    code = compile(module, filename=path, mode="exec")

    namespace = {"re": re, "datetime": datetime}
    exec(code, namespace)
    return namespace