import sqlite3
import json
import uuid
import base64
import io
import pypdf
import os
import  asyncio
from datetime import datetime
from dotenv import load_dotenv
import threading
from sympy import sympify
import re
from urllib.parse import parse_qs
from rest_framework_simplejwt.tokens import AccessToken
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async


# LangChain
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi


# LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ====================== LLMS ======================
light_1_deepseek = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="meta-llama/llama-3.3-70b-instruct:free", 
    temperature=0,
)

vision_llm_direct = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_retries=1,
)

light_3_groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY_1"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_retries=1,
)

light_4_openai_oss = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemma-4-26b-a4b:free",
    temperature=0,
)

light_5_llama_or = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0,
)

light_llm = light_3_groq.with_fallbacks(
    [light_5_llama_or, light_4_openai_oss, vision_llm_direct, light_1_deepseek]
)

# ==================== heavy llms ====================
heavy_1_gemma = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemma-4-31b-it:free",
    temperature=0,
)

heavy_2_groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY_2"),
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=1,
)

heavy_3_gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=1.0,
    max_retries=1,
)

heavy_4_openai_oss = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-flash-exp:free",
    temperature=0,
)
heavy_deepseek = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-v4-flash:free",
    temperature=0,
)

heavy_llm = heavy_2_groq.with_fallbacks(
    [heavy_1_gemma, heavy_deepseek, heavy_4_openai_oss, heavy_3_gemini_pro]
)

# ====================== Tools ======================

@tool
def calculator(expression: str) -> str:
    """Use this tool strictly for performing mathematical calculations and expressions."""
    try:
        result = sympify(expression)
        return str(result.evalf())
    except Exception as e:
        return f"Calculation error: {e}"


search_tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=3
)

@tool
def internet_search(query: str) -> str:
    """Use this tool to search the internet for live, current information."""
    try:
        results = search_tool.invoke(query)
        if not results:
            return "No results found."
        output = "\n".join([r.get("content", "") for r in results])
        return f"Search results for ({query}):\n\n{output[:1500]}"
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def summarize_text_tool(text: str) -> str:
    """Mandatory tool to use only when the user explicitly requests a summary of a long text, article, document, or book."""
    try:
        return _summarize_large_text_sync(
            text=text,
            title="النص المرسل",
            user_request="لخص النص بالكامل بدون تجاهل الأجزاء المهمة.",
        )
    except Exception as e:
        return f"Summarization failed: {str(e)}"


@tool
def query_uploaded_pdf(query: str, config: RunnableConfig) -> str:
    """Mandatory and exclusive tool to use whenever the user asks any question regarding uploaded PDF documents, requests summaries of them, or asks if you can see the file."""
    try:
        thread_id = config["configurable"].get("thread_id")
        print(f"🔍 [PDF Tool]: Fetching attachments for session: {thread_id}")

        with sqlite3.connect("db.sqlite3", timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_name, file_content FROM thread_attachments WHERE thread_id = ? AND file_type = 'pdf' ORDER BY uploaded_at DESC",
                (str(thread_id),),
            )
            rows = cursor.fetchall()

        if not rows:
            return "System Notification: No uploaded PDF files were found in the database for this session yet."

        all_text = ""
        for row in rows:
            file_name = row[0]
            file_content = row[1]
            if isinstance(file_content, bytes):
                file_content = file_content.decode("utf-8", errors="ignore")
            all_text += (
                f"\n--- Extracted Content from ({file_name}) ---\n{file_content}"
            )

        if _is_summary_request(query):
            return _summarize_large_text_sync(
                text=all_text,
                title="الملفات المرفوعة",
                user_request=query,
            )

        answer = _answer_pdf_question_sync(
            text=all_text,
            title="الملفات المرفوعة",
            question=query,
        )
        return f"Answer from uploaded documents:\n\n{answer}"
    except Exception as e:
        return f"An error occurred while attempting to parse the PDF: {str(e)}"


@tool
def analyze_uploaded_image(query: str, config: RunnableConfig) -> str:
    """Mandatory and exclusive tool to use whenever the user asks any question regarding an image, screenshot, or explicitly asks 'Can you see this image?'."""
    try:
        from langchain_core.messages import HumanMessage

        thread_id = config["configurable"].get("thread_id")
        print(f"📸 [Image Tool]: Pulling latest image for session: {thread_id}")

        with sqlite3.connect("db.sqlite3", timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT file_name, file_content FROM thread_attachments 
                WHERE thread_id = ? AND file_type = 'image' 
                ORDER BY uploaded_at DESC LIMIT 1
            """,
                (str(thread_id),),
            )
            row = cursor.fetchone()

        if not row:
            return "System Notification: No uploaded images were found in the database for this session currently."

        file_name, file_content_b64 = row

        if isinstance(file_content_b64, bytes):
            base64_str = file_content_b64.decode("utf-8")
        else:
            base64_str = str(file_content_b64)

        if "data:image" in base64_str:
            base64_str = base64_str.split(",")[-1]

        ext = "png" if str(file_name).lower().endswith("png") else "jpeg"
        mime_type = f"image/{ext}"

        vision_model =  vision_llm_direct

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Analyze the attached image and address the user's inquiry accurately. Maintain a natural, conversational tone in the same language as the user.\nUser Query: {query}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_str}"
                    },
                },
            ]
        )

        response = vision_model.invoke([message])
        return f"Image Analysis Response [{file_name}]:\n{response.content}"
    except Exception as e:
        return f"Failed to execute image analysis programmatically due to: {str(e)}"


@tool
def analyze_youtube_video(youtube_url: str, query: str) -> str:
    """Exclusive tool to use ONLY when the user provides a YouTube video URL/Link.
    You MUST pass both the 'youtube_url' and the user's 'query' to this tool to extract the right context."""
    try:
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', youtube_url)
        if not video_id_match:
            return "Error: Invalid YouTube URL format. Please provide a valid link."
        
        video_id = video_id_match.group(1)
        clean_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id, languages=["ar", "en"])
            full_transcript = " ".join([t.text for t in transcript_list]) 
        except Exception as cloud_err:
            print(f"⚠️ YouTube IP Blocked on Server: {cloud_err}")
            return "عذراً يا غالي، يوتيوب يفرض قيوداً على خوادم الاستضافة..."

        if not full_transcript or len(full_transcript.strip()) == 0:
            return "Error: Could not retrieve a text transcript."

        MAX_CHARS = 12000
        if len(full_transcript) <= MAX_CHARS:
            return f"Successfully retrieved full video transcript:\n\n{full_transcript}"

        video_summary = _summarize_large_text_sync(
            text=full_transcript,
            title=f"YouTube video {video_id}",
            user_request=query or "لخص الفيديو بالكامل بدون تجاهل الأجزاء المهمة.",
        )
        return f"The video is long, so I processed its transcript in chunks and summarized the full transcript:\n\n{video_summary}"

    except Exception as e:
        return f"Failed to programmatically load the YouTube video transcript. Error detail: {str(e)}"
    

tools = [
    internet_search,
    calculator,
    summarize_text_tool,
    query_uploaded_pdf,
    analyze_uploaded_image,
    analyze_youtube_video,
]

# ====================== Prompt ======================
system_prompt = """
أنتِ اسمك سالي — AI Assistant شخصية طبيعية جدًا، ذكية، خفيفة، وبتتكلمي بالمصري بشكل conversational.

هويتك:
- تتكلمي كإنسانة طبيعية مش روبوت.
- ردودك تكون friendly ومرنة وذكية.
- متبالغيش في الرسمية أو التنظيم إلا لو المستخدم طلب شرح تقني.
- متستخدميش markdown الكتير إلا وقت الشرح الطويل.
- استخدمي humor خفيف طبيعي أحيانًا.
- خلي الردود قصيرة نسبيًا في الشات العادي.

أسلوب الكلام:
- لو المستخدم مصري → ردي بالمصري الطبيعي.
- لو المستخدم إنجليزي → ردي إنجليزي طبيعي.
- حافظي على أسماء التقنيات والمصطلحات بالإنجليزية زي:
Python, Django, APIs, SQL, Docker, JavaScript

مهم جدًا:
- متقوليش إنك model أو AI من OpenAI أو Google أو Meta.
- لو حد سأل مين عملك:
قولي:
"أنا المساعد الذكي الخاص بالمهندس محمود البديوي ✨"

السلوك:
- متكرريش نفسك.
- مترديش بردود محفوظة.
- متديش lists كتير إلا لو مطلوبة.
- متقوليش:
"بناءً على..."
"أفهم شعورك..."
"كذكاء اصطناعي..."
"سأقوم بتحليل..."
- جاوبي مباشرة.

━━━━━━━━━━━━━━
TOOLS RULES
━━━━━━━━━━━━━━

عندك أدوات قوية — استخدميها تلقائيًا وقت الحاجة فقط.

1) calculator
استخدميها للحسابات والرياضيات فقط.

2) internet_search
استخدميها فقط للمعلومات الحالية أو الأخبار أو الأسعار أو أي حاجة محتاجة تحديث مباشر.

أمثلة:
- مين رئيس أمريكا دلوقتي
- سعر الدولار
- آخر أخبار برشلونة

ممنوع تألف معلومات حالية بدون بحث.

3) query_uploaded_pdf
استخدميها فقط لو المستخدم بيسأل عن:
- PDF
- CV
- ملف مرفوع
- "اقرأ الملف"
- "لخص الملف"
- "ايه اللي في السيفي"

4) analyze_uploaded_image
استخدميها فقط لو المستخدم بيتكلم عن:
- صورة
- screenshot
- "شايف الصورة؟"
- "حلل الصورة"

5) analyze_youtube_video
استخدميها فقط لو المستخدم أرسل YouTube link أو طلب تلخيص فيديو.

━━━━━━━━━━━━━━
IMPORTANT BEHAVIOR
━━━━━━━━━━━━━━

- الشات العادي يكون conversational بدون tools.
- متدخليش agent mode إلا لو فعلًا محتاج tool.
- لو السؤال بسيط جاوبي ببساطة.
- لو السؤال تقني اشرحي بهدوء وترتيب.
- لو المستخدم بيهزر اهزري معاه بشكل طبيعي.
- متبقيش verbose بدون داعي.

━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━

User: اسمك اي
Assistant: أنا سالي ✨

User: بتعرفي تعملي اي
Assistant:
أي حاجة تقريبًا 😂
هات كود، CV، PDF، صورة، مشكلة مذاكرة، فكرة مشروع وأنا أظبطهالك.

User: عندي error في Django
Assistant:
ابعت الـ error أو الكود وأنا أمشيهالك واحدة واحدة.

User: لخص الملف ده
Assistant:
[تستخدم query_uploaded_pdf]

User: شايف الصورة؟
Assistant:
[تستخدم analyze_uploaded_image]

Today's date: {current_date}
"""


_HEAVY_KEYWORDS = {
    "كود", "برمج", "برمجة", "code", "python", "django", "sql", "api", "خوارزمية", 
    "error", "bug", "class", "function", "pdf", "ملف", "صورة", "صوره", "screenshot", 
    "لقطة", "لخص", "لخصلي", "تلخيص", "حلل", "تحليل", "قارن", "مقارنة", "تقرير", 
    "احسب", "حساب", "معادلة", "رياضيات", "رئيس", "ملك", "حاكم", "وزير", "دولة", 
    "عمر", "سن", "من هو", "مين", "كم", "يوتيوب", "فيديو", "youtube", "video", 
    "رابط", "لينك", "link"
}


def _route_message(msg: str) -> str:
    msg_lower = msg.lower().strip()
    if "youtu.be" in msg_lower or "youtube.com" in msg_lower:
        return "HEAVY"
    if re.search(r"[\u0600-\u06FF]", msg_lower):
        return "HEAVY"
    if re.match(r"^[a-zA-Z\s\d\W;]+$", msg_lower):
        if not any(kw in msg_lower for kw in {"code", "python", "hello", "hi"}):
            return "HEAVY"
    if any(kw in msg_lower for kw in _HEAVY_KEYWORDS):
        return "HEAVY"
    return "LIGHT"

async def stream_llm_async(llm, prompt):
    """بتشغّل الـ sync LLM stream في thread منفصل وبترجع chunks"""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def run_stream():
        try:
            for chunk in llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.content)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=run_stream, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item



_INTERNAL_LEAK_PATTERNS = (
    r"\[User Query Analysis\]",
    r"Based on the user",
    r"Based on your",
    r"Based on previous",
    r"Based on .*profile",
    r"The user's query is",
    r"which translates to",
    r"I will (try|attempt|analyze|ask)",
    r"I'll (try|attempt|analyze|ask)",
    r"I am assuming",
    r"assuming you'?re asking",
    r"To confirm, I'll ask",
    r"To clarify",
    r"Do you know what I'm trying to ask",
    r"Considering the user's",
    r"current query",
    r"President of Egypt",
)


def _contains_internal_leak(text: str) -> bool:
    if not text:
        return False
    return any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in _INTERNAL_LEAK_PATTERNS)

def _normalize_arabic_query(text: str) -> str:
    text = str(text or "").lower().strip()
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
        "ؤ": "و", "ئ": "ي",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return text


def _is_capability_question(message: str) -> bool:
    normalized = _normalize_arabic_query(message)
    compact = re.sub(r"\s+", "", normalized)
    english = str(message or "").lower()
    arabic_hits = (
        "بتعرفتعمل" in compact
        or "تعرفتعمل" in compact
        or "تقدرتعمل" in compact
        or "تعمل اي" in normalized
        or "تعمل ايه" in normalized
        or "اي قدراتك" in normalized
        or "كايجنت" in compact
        or "كذكاء" in compact
        or "كمساعد" in compact
    )
    english_hits = any(phrase in english for phrase in (
        "what can you do", "what do you do", "as an agent", "as ai", "as an ai"
    ))
    return arabic_hits or english_hits


def _capability_reply() -> str:
    return """أيوه يا صاحبي، اعتبرني AI agent معاك في الشات. أقدر أساعدك في حاجات كتير، زي:

- أظبطلك كود وأشرح errors في Python, Django, JavaScript, SQL, APIs.
- أقرأ PDF أو CV وأجاوبك من الملف نفسه.
- أحلل screenshots أو صور وتقولّي عايز تفهم إيه فيها.
- ألخص YouTube links أو مقالات أو notes.
- أعملك خطة مذاكرة، roadmap، أفكار مشاريع، أو أرتبلك يومك.
- أساعدك تكتب emails، prompts، بوستات، رسائل، أو تحسن CV و LinkedIn.
- أبحثلك عن معلومات حديثة من الإنترنت لما الموضوع محتاج تحديث.

كلمني عادي جدًا: "اشرحلي ده"، "ظبط الكود"، "اقرأ الملف"، "اعمللي خطة"، وأنا أمشي معاك خطوة خطوة."""

def _is_pdf_related_question(message: str) -> bool:
    normalized = _normalize_arabic_query(message)
    compact = re.sub(r"\s+", "", normalized)
    english = str(message or "").lower()

    arabic_hits = any(term in normalized for term in (
        "سيفي", "السيفي", "سي في", "السيره", "السيرة", "الملف المرفوع", "المرفق",
        "بي دي اف", "اعرفني", "عرفني", "معلومات عني", "كلمك عني", "كلمني عني",
        "انا مين", "من انا", "انا اللي", "انا اللى", "في الملف", "من الملف",
        "الملف ده", "الملف دا", "الملف دة"
    ))
    compact_hits = any(term in compact for term in (
        "السيفي", "سيفي", "اعرفني", "عرفني", "انااللي", "انااللى", "منانا"
    ))
    english_hits = any(term in english for term in (
        "cv", "resume", "pdf", "document", "my profile", "about me"
    ))
    return arabic_hits or compact_hits or english_hits

def _needs_live_search(message: str) -> bool:
    normalized = _normalize_arabic_query(message)
    compact = re.sub(r"\s+", "", normalized)
    english = str(message or "").lower()

    arabic_live_terms = (
        "حاليا", "دلوقتي", "الان", "النهارده", "اليوم", "اخر", "احدث",
        "اخبار", "سعر", "اسعار", "نتيجه", "نتيجة", "مباشر"
    )
    arabic_position_terms = (
        "رئيس", "الرئيس", "ملك", "الملك", "وزير", "الوزير", "حاكم", "الحاكم",
        "رئيس الوزراء", "محافظ", "الرئيس الحالي"
    )
    arabic_question_terms = ("مين", "من", "من هو", "من هي", "ايه", "ما هو", "ما هي")

    has_position_question = any(q in normalized for q in arabic_question_terms) and any(
        term in normalized for term in arabic_position_terms
    )
    has_explicit_live = any(term in normalized for term in arabic_live_terms)
    has_common_president_query = any(term in compact for term in (
        "مينرئيس", "منرئيس", "رئيسمصر", "رئيسامريكا", "رئيسالولاياتالمتحده",
        "الرئيسالحالي", "رئيسفرنسا", "رئيسروسيا"
    ))

    english_live = any(term in english for term in (
        "current", "today", "latest", "news", "price", "score", "now", "live"
    ))
    english_position = any(term in english for term in (
        "president", "prime minister", "king", "minister", "governor", "ceo", "champion"
    ))
    english_question = any(term in english for term in ("who", "what", "which"))

    return has_common_president_query or has_position_question or (has_explicit_live and any(
        term in normalized for term in arabic_position_terms
    )) or (english_position and (english_question or english_live))


def _search_query_for_user_message(message: str) -> str:
    normalized = _normalize_arabic_query(message)
    compact = re.sub(r"\s+", "", normalized)
    if "رئيسامريكا" in compact or "رئيسالولاياتالمتحده" in compact:
        return "current president of the United States official"
    if "رئيسمصر" in compact:
        return "الرئيس الحالي لجمهورية مصر العربية الموقع الرسمي"
    return str(message or "").strip()


def _split_text_chunks(text: str, chunk_size: int = 5500, overlap: int = 550):
    text = str(text or "").strip()
    if not text:
        return []
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n--- Page ", "\n\n", "\n", ". ", "؟ ", "! ", " ", ""],
    ).split_text(text)


def _is_summary_request(message: str) -> bool:
    normalized = _normalize_arabic_query(message)
    english = str(message or "").lower()
    arabic_terms = (
        "لخص", "تلخيص", "ملخص", "اختصر", "الخلاصه", "الخلاصة",
        "النقاط المهمه", "النقاط المهمة", "اهم الافكار", "اهم النقاط",
        "لخصلي", "شرح الكتاب", "ملخص الكتاب"
    )
    english_terms = ("summary", "summarize", "summarise", "recap", "key points")
    return any(term in normalized for term in arabic_terms) or any(
        term in english for term in english_terms
    )


def _is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


def _llm_text(response) -> str:
    return str(response.content if hasattr(response, "content") else response).strip()


def _summarize_large_text_sync(text: str, title: str = "", user_request: str = "") -> str:
    chunks = _split_text_chunks(text, chunk_size=6000, overlap=650)
    if not chunks:
        return "مش لاقي نص واضح أقدر ألخصه."

    wants_arabic = _is_arabic_text(user_request) or _is_arabic_text(text[:1000])
    language_rule = (
        "اكتب بالعربية الطبيعية الودودة، وحافظ على English terms كما هي."
        if wants_arabic
        else "Write in natural English and preserve original names and terms."
    )

    partial_summaries = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_prompt = f"""You are summarizing a large document in a careful map-reduce workflow.

Document title: {title or "Untitled"}
User request: {user_request or "Summarize the whole document."}
Chunk: {idx}/{total}

Rules:
- Do not ignore details just because this is one part of a bigger document.
- Extract the important ideas, facts, names, dates, numbers, definitions, arguments, examples, and action items from this chunk.
- Preserve English names/terms exactly as written.
- If the chunk is from a book, capture chapter/section logic when visible.
- {language_rule}

Chunk text:
{chunk}

Chunk summary:"""
        partial_summaries.append(_llm_text(heavy_llm.invoke(chunk_prompt)))

    grouped = partial_summaries
    round_no = 1
    while len("\n\n".join(grouped)) > 24000 or len(grouped) > 10:
        next_grouped = []
        for start in range(0, len(grouped), 8):
            group = "\n\n".join(
                f"Part summary {start + offset + 1}:\n{summary}"
                for offset, summary in enumerate(grouped[start:start + 8])
            )
            reduce_prompt = f"""Compress these partial summaries without losing important information.

Document title: {title or "Untitled"}
Reduction round: {round_no}
{language_rule}

Rules:
- Merge repeated points.
- Keep names, numbers, dates, technical terms, conclusions, and examples.
- Do not invent anything.

Partial summaries:
{group}

Merged summary:"""
            next_grouped.append(_llm_text(heavy_llm.invoke(reduce_prompt)))
        grouped = next_grouped
        round_no += 1

    combined = "\n\n".join(
        f"Part {idx} summary:\n{summary}"
        for idx, summary in enumerate(grouped, start=1)
    )
    final_prompt = f"""Create the final high-quality summary from these complete partial summaries.

Document title: {title or "Untitled"}
User request: {user_request or "Summarize the document."}
{language_rule}

Output style:
- Start with a short direct overview.
- Then give organized sections.
- Include main ideas, important details, names, dates, numbers, examples, and conclusions.
- If the document is long, make the summary rich enough to be useful, not tiny.
- Mention if the source text appears incomplete or extraction quality is weak.
- Do not say you only saw the beginning/middle/end; you processed chunk summaries from the whole text.

Partial summaries from the whole document:
{combined}

Final summary:"""
    return _llm_text(heavy_llm.invoke(final_prompt))


def _query_terms(text: str):
    normalized = _normalize_arabic_query(text)
    tokens = re.findall(r"[\w\u0600-\u06FF]+", normalized.lower())
    stopwords = {
        "اي", "ايه", "ما", "من", "هو", "هي", "في", "عن", "على", "انا", "انت",
        "ده", "دا", "دي", "اللي", "اللى", "بتاع", "بتاعي", "قولي", "قولى",
        "what", "who", "is", "the", "a", "an", "of", "in", "to", "me", "my",
    }
    return [token for token in tokens if len(token) > 1 and token not in stopwords]


def _select_relevant_text_context(text: str, question: str, max_chunks: int = 8) -> str:
    chunks = _split_text_chunks(text, chunk_size=4200, overlap=450)
    if not chunks:
        return ""
    if len(text) <= 18000:
        return text

    terms = _query_terms(question)
    scored = []
    for idx, chunk in enumerate(chunks):
        searchable = _normalize_arabic_query(chunk).lower()
        score = sum(searchable.count(term) for term in terms)
        scored.append((score, idx, chunk))

    selected_indexes = {0}
    if len(chunks) > 1:
        selected_indexes.add(len(chunks) - 1)

    for score, idx, _chunk in sorted(scored, key=lambda item: item[0], reverse=True):
        if score <= 0 and len(selected_indexes) >= 3:
            break
        selected_indexes.add(idx)
        if len(selected_indexes) >= max_chunks:
            break

    return "\n\n".join(
        f"--- Relevant excerpt {idx + 1}/{len(chunks)} ---\n{chunks[idx]}"
        for idx in sorted(selected_indexes)
    )


def _answer_pdf_question_sync(text: str, title: str, question: str) -> str:
    if _is_summary_request(question):
        return _summarize_large_text_sync(text=text, title=title, user_request=question)

    context = _select_relevant_text_context(text, question)
    if not context:
        return "مش لاقي نص واضح في الملف أقدر أجاوب منه."

    wants_arabic = _is_arabic_text(question)
    language_rule = (
        "جاوب بالعربية الطبيعية الودودة، وحافظ على English terms كما هي."
        if wants_arabic
        else "Answer in natural English and preserve original terms."
    )
    answer_prompt = f"""Answer the user's question using the provided PDF/document context.

Document title: {title}
User question: {question}

Rules:
- {language_rule}
- Use only the provided document context.
- If the user is asking about themselves after uploading a CV/resume, treat the CV owner as the user.
- The Arabic word "السيفي" means CV/resume, not sword.
- Be helpful and specific. Include names, skills, dates, numbers, and examples when present.
- If the answer is not in the selected context, say that clearly and suggest asking for a full summary.

Document context:
{context}

Answer:"""
    return _llm_text(heavy_llm.invoke(answer_prompt))


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):

        await self.accept()
        self.last_youtube_transcript = ""
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")
        formatted_system_prompt = system_prompt.format(current_date=current_date_str)

        self.db = "db.sqlite3"
        self.conn = sqlite3.connect(self.db, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thread_attachments (
                file_id TEXT PRIMARY KEY,
                thread_id TEXT,
                file_name TEXT,
                file_content TEXT,
                file_type TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                thread_id TEXT PRIMARY KEY,
                facts TEXT DEFAULT ''
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                role TEXT,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        self.memory = SqliteSaver(self.conn)
        self.memory.setup()

        self.heavy_agent = create_react_agent(
            heavy_llm, tools, checkpointer=self.memory, prompt=formatted_system_prompt
        )
        self.light_agent = create_react_agent(
            light_llm, tools, checkpointer=self.memory, prompt=formatted_system_prompt
        )
        self.fallback_agent = create_react_agent(
            light_1_deepseek,
            tools,
            checkpointer=self.memory,
            prompt=formatted_system_prompt,
        )

        query_params = parse_qs(self.scope.get("query_string", b"").decode())
        token = query_params.get("token", [None])[0]
        guest_id = query_params.get("guest_id", [None])[0]

        token_user_id = None
        if token:
            try:
                token_user_id = AccessToken(token).get("user_id")
            except Exception:
                token_user_id = None

        user = self.scope.get("user")
        if user and user.is_authenticated:
            self.thread_id = f"user_session_{user.id}"
        elif token_user_id:
            self.thread_id = f"user_session_{token_user_id}"
        elif guest_id:
            self.thread_id = f"guest_session_{guest_id}"
        else:
            self.thread_id = f"guest_session_{uuid.uuid4().hex[:4]}"

        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": 25,
        }
        try:
            chat_history = await self.load_chat_history()
            if chat_history:
                await self.send(text_data=json.dumps({
                    "type": "history",
                    "messages": chat_history
                }))
        except Exception as e:
            print(f"❌ فشل في تحميل تاريخ الشات: {e}")

    @database_sync_to_async
    def save_chat_message(self, role, message):
        if not message or not str(message).strip():
            return

        with sqlite3.connect(self.db, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_messages (id, thread_id, role, message)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), self.thread_id, role, str(message)),
            )
            conn.commit()


    @database_sync_to_async
    def load_chat_history(self):
        with sqlite3.connect(self.db, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, message FROM chat_messages
                WHERE thread_id = ?
                ORDER BY created_at ASC
                """,
                (self.thread_id,),
            )
            history = []
            for role, message in cursor.fetchall():
                if role == "bot" and _contains_internal_leak(message):
                    continue
                history.append({"role": role, "message": message})
            return history

    async def _send_text_chunks(self, text: str, chunk_size: int = 900):
        text = str(text or "").strip()
        if not text:
            text = "تمام يا صاحبي، معاك. ابعتلي اللي محتاجه وأنا أساعدك خطوة بخطوة."
        for start in range(0, len(text), chunk_size):
            await self.send(text_data=json.dumps({
                "type": "stream_chunk",
                "chunk": text[start:start + chunk_size]
            }))

    async def _prepare_bot_reply(self, reply: str, user_message: str) -> str:
        reply = str(reply or "").strip()
        if reply and not _contains_internal_leak(reply):
            return reply

        repair_prompt = f"""
Rewrite the assistant answer so it is a clean final reply to the user only.

Rules:
- Reply in the same language and dialect as the user. If Arabic/Egyptian Arabic, be warm, friendly, and direct.
- Do not include analysis, translation of the user's message, assumptions, hidden reasoning, labels, or phrases like "Based on".
- Preserve English technical terms/names exactly as written.
- If the old answer asked an unnecessary clarifying question, answer the likely intent directly when possible.

User message:
{user_message}

Bad assistant answer:
{reply}

Clean final answer:
"""
        try:
            fixed = await asyncio.to_thread(light_llm.invoke, repair_prompt)
            fixed_text = fixed.content if hasattr(fixed, "content") else str(fixed)
            fixed_text = str(fixed_text or "").strip()
            if fixed_text and not _contains_internal_leak(fixed_text):
                return fixed_text
        except Exception as repair_err:
            print(f"Reply repair failed: {repair_err}")

        return "تمام يا صاحبي، فهمتك. قولّي محتاج تعرف إيه بالظبط وأنا هجاوبك بشكل مباشر وبسيط."

    async def _answer_with_live_search(self, message: str) -> str:
        query = _search_query_for_user_message(message)
        try:
            search_output = await asyncio.to_thread(internet_search.invoke, {"query": query})
        except Exception as first_err:
            try:
                search_output = await asyncio.to_thread(internet_search.invoke, query)
            except Exception as second_err:
                print(f"Live search failed: {first_err} / {second_err}")
                return "مش هفتي عليك يا صاحبي. حاولت أتحقق من المعلومة الحالية بس البحث فشل عندي مؤقتًا. جرّب تاني بعد لحظات أو ابعتلي صياغة أدق وأنا أتحقق لك."

        answer_prompt = f"""أنت مساعد شخصي عالمي وودود. جاوب المستخدم من نتائج البحث فقط.

قواعد مهمة:
- جاوب بنفس لغة المستخدم ولهجته. لو عربي مصري، خليك طبيعي وودود.
- ابدأ بالإجابة المباشرة، ثم أضف سياقًا قصيرًا مفيدًا.
- لا تقل "لا أعرف" طالما نتائج البحث فيها معلومة كافية.
- لو نتائج البحث غير كافية أو متضاربة، قل بوضوح إنك لم تجد تأكيدًا كافيًا.
- حافظ على أسماء الأشخاص والدول والمصطلحات كما هي.
- لا تعرض تحليل داخلي.

سؤال المستخدم:
{message}

نتائج البحث:
{search_output}

الإجابة:"""
        raw_content = ""
        async for text in stream_llm_async(heavy_llm, answer_prompt):
            raw_content += text
        return await self._prepare_bot_reply(raw_content, message)
    @database_sync_to_async
    def update_user_memories(self, new_messages_text):
        try:
            if not str(new_messages_text or "").startswith("User uploaded PDF:"):
                return

            with sqlite3.connect(self.db, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT facts FROM user_memories WHERE thread_id = ?", (self.thread_id,)
                )
                row = cursor.fetchone()
                current_facts = row[0].strip() if row and row[0] else ""
                new_fact = str(new_messages_text).strip()[:4000]
                updated_facts = f"{current_facts}\n\n{new_fact}".strip() if current_facts else new_fact
                cursor.execute(
                    "REPLACE INTO user_memories (thread_id, facts) VALUES (?, ?)",
                    (self.thread_id, updated_facts),
                )
                conn.commit()
        except Exception as me:
            print(f"Memory update failed: {me}")

    def _get_recent_file_context(self) -> str:
        context = ""
        try:
            with sqlite3.connect(self.db, timeout=20) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_name, file_content FROM thread_attachments
                    WHERE thread_id = ? AND file_type = 'pdf'
                    ORDER BY uploaded_at DESC LIMIT 1
                """, (self.thread_id,))
                pdf = cursor.fetchone()
                if pdf:
                    file_name, file_content = pdf
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode("utf-8", errors="ignore")
                    file_content = str(file_content or "").strip()
                    MAX_CHARS = 8000
                    if len(file_content) > MAX_CHARS:
                        segment_size = 1400
                        starts = [
                            0,
                            max(0, len(file_content) // 4 - segment_size // 2),
                            max(0, len(file_content) // 2 - segment_size // 2),
                            max(0, (len(file_content) * 3) // 4 - segment_size // 2),
                            max(0, len(file_content) - segment_size),
                        ]
                        seen = set()
                        sampled_parts = []
                        for part_number, start in enumerate(starts, start=1):
                            if start in seen:
                                continue
                            seen.add(start)
                            sampled_parts.append(
                                f"--- Distributed excerpt {part_number} ---\n"
                                f"{file_content[start:start + segment_size]}"
                            )
                        file_content = "\n\n".join(sampled_parts)
                    context += f"""
                        [Uploaded PDF: {file_name}]
                        {file_content}
                        Instruction: Answer directly from this PDF if the question is related to it.
                        """
                cursor.execute("""
                    SELECT file_name FROM thread_attachments
                    WHERE thread_id = ? AND file_type = 'image'
                    ORDER BY uploaded_at DESC LIMIT 1
                """, (self.thread_id,))
                img = cursor.fetchone()
                if img:
                    context += f"\n[System Notice: Image uploaded: '{img[0]}'. Use analyze_uploaded_image tool if asked.]\n"
        except Exception as e:
            print(f"File context injection failed: {e}")
        return context

    def _get_latest_pdf_payload(self):
        try:
            with sqlite3.connect(self.db, timeout=20) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_name, file_content FROM thread_attachments
                    WHERE thread_id = ? AND file_type = 'pdf'
                    ORDER BY uploaded_at DESC LIMIT 1
                """, (self.thread_id,))
                row = cursor.fetchone()
            if not row:
                return None
            file_name, file_content = row
            if isinstance(file_content, bytes):
                file_content = file_content.decode("utf-8", errors="ignore")
            file_content = str(file_content or "").strip()
            if not file_content:
                return None
            return file_name, file_content
        except Exception as e:
            print(f"Latest PDF fetch failed: {e}")
            return None

    async def disconnect(self, close_code):
        if hasattr(self, "conn"):
            self.conn.close()

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        msg_type = text_data_json.get("type", "text")

        user_message_check = text_data_json.get("message", "")
        is_english = any(ord(char) < 128 for char in user_message_check if char.isalpha())

        # ==========================================
        #                   PDF
        # ==========================================
        if msg_type == "file":
            try:
                file_name = text_data_json["file_name"]
                file_data_b64 = text_data_json["file_data"]
                file_bytes = base64.b64decode(file_data_b64)
                pdf_file = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(pdf_file)

                extracted_text = ""
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        extracted_text += f"\n\n--- Page {page_number} ---\n{text.strip()}\n"

                with sqlite3.connect(self.db, timeout=30) as conn:
                    cursor = conn.cursor()
                    unique_file_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO thread_attachments (file_id, thread_id, file_name, file_content, file_type)
                        VALUES (?, ?, ?, ?, 'pdf')
                    """,
                        (unique_file_id, self.thread_id, file_name, extracted_text),
                    )
                    conn.commit()

                bot_reply = ""
                await self.send(text_data=json.dumps({"type": "stream_start"}))

                summary_request = (
                    f"المستخدم رفع ملف PDF اسمه '{file_name}'. "
                    "قدم ملخص شامل ومفيد يغطي كل أجزاء الملف، وليس أول/نصف/آخر الملف فقط. "
                    "حافظ على أي English terms كما هي."
                )
                bot_reply = await asyncio.to_thread(
                    _summarize_large_text_sync,
                    extracted_text,
                    file_name,
                    summary_request,
                )

                bot_reply = await self._prepare_bot_reply(bot_reply, f"رفع PDF: {file_name}")
                await self._send_text_chunks(bot_reply)
                await self.send(text_data=json.dumps({"type": "stream_end"}))
                from langchain_core.messages import HumanMessage, AIMessage
                
                new_messages = [
                    HumanMessage(content=f"لقد قمت برفع ملف PDF اسمه '{file_name}'. محتواه تم حفظه في قاعدة البيانات بنجاح، يمكنك قراءته عند سؤالي عنه باستخدام أداة query_uploaded_pdf."),
                    AIMessage(content=bot_reply)
                ]
                
                self.heavy_agent.update_state(self.config, {"messages": new_messages})
                
                await self.update_user_memories(f"User uploaded PDF: {file_name}\nBot Summary: {bot_reply}")
                await self.save_chat_message("user", f"تم رفع ملف PDF: {file_name}")
                await self.save_chat_message("bot", bot_reply)
                
                return 

            except Exception as e:
                print(f"❌ Error in file upload: {e}")
                await self.send(text_data=json.dumps({"type": "stream_start"}))
                await self.send(text_data=json.dumps({"type": "stream_chunk", "chunk": "حدث خطأ أثناء معالجة الملف، يرجى إعادة المحاولة."}))
                await self.send(text_data=json.dumps({"type": "stream_end"}))
                return
        # ==========================================
        #                   images
        # ==========================================
        if msg_type == "image":
            try:
                fileName = text_data_json["file_name"]
                file_data = text_data_json["file_data"]

                if isinstance(file_data, str) and not file_data.startswith("data:image"):
                    image_b64_to_save = file_data
                elif isinstance(file_data, str) and file_data.startswith("data:image"):
                    image_b64_to_save = file_data.split(",")[1]
                else:
                    image_b64_to_save = base64.b64encode(file_data).decode("utf-8")

                with sqlite3.connect(self.db, timeout=30) as conn:
                    cursor = conn.cursor()
                    unique_file_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO thread_attachments (file_id, thread_id, file_name, file_content, file_type)
                        VALUES (?, ?, ?, ?, 'image')
                    """,
                        (unique_file_id, self.thread_id, fileName, image_b64_to_save),
                    )
                    conn.commit()
                bot_reply = (
                    f"Successfully received the image '{fileName}'."
                    if is_english
                    else f"تم استلام صورة '{fileName}' بنجاح وعيوني شيفاها دلوقتي، اسألني عنها في أي وقت!"
                )
            except Exception as e:
                bot_reply = "An error occurred" if is_english else "حدث خطأ أثناء استقبال الصورة."
            await self.save_chat_message("user", f"تم رفع صورة: {fileName}")
            await self.save_chat_message("bot", bot_reply)
            await self.send(text_data=json.dumps({"reply": bot_reply}))
            return

        message = text_data_json.get("message", "")
        await self.save_chat_message("user", message)

        if _is_capability_question(message):
            bot_reply = _capability_reply()
            await self.send(text_data=json.dumps({"type": "stream_start"}))
            await self._send_text_chunks(bot_reply)
            await self.send(text_data=json.dumps({"type": "stream_end"}))
            await self.save_chat_message("bot", bot_reply)
            return

        if _needs_live_search(message):
            await self.send(text_data=json.dumps({"type": "stream_start"}))
            bot_reply = await self._answer_with_live_search(message)
            await self._send_text_chunks(bot_reply)
            await self.send(text_data=json.dumps({"type": "stream_end"}))
            await self.save_chat_message("bot", bot_reply)
            return

        latest_pdf = self._get_latest_pdf_payload()
        if latest_pdf and (_is_pdf_related_question(message) or _is_summary_request(message)):
            file_name, pdf_text = latest_pdf
            await self.send(text_data=json.dumps({"type": "stream_start"}))
            bot_reply = await asyncio.to_thread(
                _answer_pdf_question_sync,
                pdf_text,
                file_name,
                message,
            )
            bot_reply = await self._prepare_bot_reply(bot_reply, message)
            await self._send_text_chunks(bot_reply)
            await self.send(text_data=json.dumps({"type": "stream_end"}))
            await self.save_chat_message("bot", bot_reply)
            await self.update_user_memories(f"User uploaded PDF: {file_name}\nBot Summary: {bot_reply}")
            return

        try:
            with sqlite3.connect(self.db, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT facts FROM user_memories WHERE thread_id = ?", (self.thread_id,))
                row = cursor.fetchone()
                user_facts = row[0] if row else "No historic context data found yet."

                cursor.execute(
                    """
                    SELECT file_name, file_content FROM thread_attachments 
                    WHERE thread_id = ? AND file_type = 'image' 
                    ORDER BY uploaded_at DESC LIMIT 1
                """,
                    (self.thread_id,),
                )
                image_row = cursor.fetchone()

            image_keywords = {"صوره", "صورة", "screenshot", "لقطة", "شايف", "دي", "المنشور", "image", "pic", "حل", "اشرح"}
            is_asking_about_image = any(kw in message.lower() for kw in image_keywords)

            if image_row and is_asking_about_image:
                file_name, base64_str = image_row
                if isinstance(base64_str, bytes):
                    base64_str = base64_str.decode("utf-8")
                if "data:image" in base64_str:
                    base64_str = base64_str.split(",")[-1]
                    
                ext = "png" if str(file_name).lower().endswith("png") else "jpeg"
                mime_type = f"image/{ext}"

                formatted_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Answer the user naturally about the attached image. User query: {message}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_str}"}
                            }
                        ]
                    }
                ]
                
                try:
                    response = await asyncio.to_thread(vision_llm_direct.invoke, formatted_messages)
                    raw_content = response.content
                except Exception as vision_err:
                    print(f"⚠️ Direct Vision Model Error: {vision_err}")
                    raw_content = "عذراً، حدث خطأ أثناء تحليل الصورة المباشر."

                raw_content = await self._prepare_bot_reply(raw_content, message)
                await self.send(text_data=json.dumps({"type": "stream_start"}))
                await self._send_text_chunks(raw_content)
                await self.send(text_data=json.dumps({"type": "stream_end"}))

            else:
                youtube_match = re.search(
                    r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]+)',
                    message
                )
                if youtube_match:
                    youtube_url = youtube_match.group(1)
                    try:
                        transcript_content = await asyncio.to_thread(
                            analyze_youtube_video.invoke, {"youtube_url": youtube_url, "query": message}
                        )
                        self.last_youtube_transcript = transcript_content

                        llm_prompt = f"""أنت مساعد ذكي محترف. المستخدم أرسل رابط يوتيوب وطلب: "{message}"
                        
                المحتوى المستخرج من الفيديو:
                {transcript_content}

                قم بالرد على طلب المستخدم بناءً على محتوى الفيديو بشكل احترافي، مباشر، ومفصل. 
                شروط هامة جداً للاستجابة:
                1. إياك وتكرار الجمل أو الفقرات (لا تدخل في حلقة مفرغة).
                2. حافظ على المصطلحات التقنية والبرمجية وأسماء التقنيات باللغة الإنجليزية كما هي، لا تقم بترجمتها للعربية.
                """
                        
                        raw_content = ""
                        await self.send(text_data=json.dumps({"type": "stream_start"}))
                        
                        async for text in stream_llm_async(heavy_llm, llm_prompt): 
                            raw_content += text
                        raw_content = await self._prepare_bot_reply(raw_content, message)
                        await self._send_text_chunks(raw_content)
                        await self.send(text_data=json.dumps({"type": "stream_end"}))
                    except Exception as yt_err:
                        await self.send(text_data=json.dumps({"type": "stream_start"}))
                        print(f"⚠️ YouTube Analysis Error: {yt_err}")
                        error_msg = "عذراً، حدث خطأ أثناء تحليل الفيديو. قد يكون الفيديو طويلاً جداً أو مقيداً."
                        raw_content = error_msg
                        await self.send(text_data=json.dumps({
                            "type": "stream_chunk",
                            "chunk": error_msg
                        }))
                        await self.send(text_data=json.dumps({"type": "stream_end"}))
                else:
                    route_decision = _route_message(message)
                    active_agent = self.heavy_agent if "HEAVY" in route_decision else self.light_agent
                    file_hint = self._get_recent_file_context()
                    
                    youtube_context = ""
                    if hasattr(self, 'last_youtube_transcript') and self.last_youtube_transcript:
                        youtube_context = f"\n[Last YouTube Video Content]:\n{self.last_youtube_transcript[:4000]}\n"
                    
                    formatted_user_message = f"""PRIVATE CONTEXT FOR ASSISTANT ONLY - do not mention or describe this context to the user.
Use recent file/video context only when the user's message is clearly about it.
Do not infer the user's intent from old memory. Do not assume a topic from previous messages.

Recent file/video context:
{file_hint}{youtube_context}

USER MESSAGE:
{message}

Reply directly to USER MESSAGE in the user's natural language and tone. Do not translate the message, do not expose analysis, and do not mention this private context unless the user explicitly asks about memory/files."""
                    messages_to_send = [("user", formatted_user_message)]

                    try:
                        raw_content = ""
                        await self.send(text_data=json.dumps({"type": "stream_start"}))

                        async for msg_event in active_agent.astream(
                            {"messages": messages_to_send},
                            config=self.config,
                            stream_mode="messages"
                        ):
                            if isinstance(msg_event, tuple):
                                chunk = msg_event[0]
                            else:
                                chunk = msg_event
                                
                            if hasattr(chunk, 'type') and chunk.type == "ai":
                                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                                    continue
                                if isinstance(chunk.content, str) and chunk.content:
                                    raw_content += chunk.content

                        bot_reply = await self._prepare_bot_reply(raw_content, message)
                        await self._send_text_chunks(bot_reply)
                        raw_content = bot_reply
                        await self.send(text_data=json.dumps({"type": "stream_end"}))
                    except Exception as agent_err:
                        print(f"⚠️ Circuit breaker triggered: {agent_err}")
                        try:
                            raw_content = ""
                            async for text in stream_llm_async(light_llm, messages_to_send):
                                raw_content += text
                            bot_reply = await self._prepare_bot_reply(raw_content, message)
                            await self._send_text_chunks(bot_reply)
                            raw_content = bot_reply
                            await self.send(text_data=json.dumps({"type": "stream_end"}))
                        except Exception:
                            raw_content = "عذراً، حدث خطأ مؤقت."
                            await self._send_text_chunks(raw_content)
                            await self.send(text_data=json.dumps({"type": "stream_end"}))
            if isinstance(raw_content, str):
                bot_reply = raw_content
            elif isinstance(raw_content, list):
                texts = [part["text"] for part in raw_content if isinstance(part, dict) and "text" in part]
                bot_reply = " ".join(texts) if texts else str(raw_content)
            else:
                bot_reply = str(raw_content)

            await self.update_user_memories(f"User: {message}\nBot: {bot_reply}")
            await self.save_chat_message("bot", bot_reply)

        except Exception as e:
            print(f"Fatal error in processing: {e}")
            bot_reply = "An error occurred with the network connection." if is_english else "عذراً يا غالي، يبدو أن هناك مشكلة اتصال عامة بالشبكة."

            await self.send(text_data=json.dumps({"reply": bot_reply}))
            return
