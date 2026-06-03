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


from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi


from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

vision_llm_direct = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,      
    max_retries=1,
)

vision_llm_chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,    
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
    [light_5_llama_or, light_4_openai_oss, vision_llm_direct]
)

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
    temperature=0,
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

_deepseek_chat = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-v4-flash:free",
    temperature=0.7,
)

_groq_chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY_1"),
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_retries=1,
)

_gemma_chat = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemma-4-31b-it:free",
    temperature=0.7,
)

simple_chat_llm = vision_llm_chat.with_fallbacks(
    [_deepseek_chat, _groq_chat, _gemma_chat]
)

search_tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=3
)

@tool
def calculator(expression: str) -> str:
    """Use this tool strictly for performing mathematical calculations and expressions."""
    try:
        result = sympify(expression)
        return str(result.evalf())
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def internet_search(query: str) -> str:
    """Use this tool to search the internet for live, current information."""
    try:
        results = search_tool.invoke(query)
        if not results:
            return "No results found."
        output = "\n".join([r.get("content", "") for r in results])
        return f"Search results for ({query}):\n\n{output[:3000]}"
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
    """استخدم هذه الأداة للبحث عن معلومات داخل ملف الـ PDF الذي قام المستخدم برفعه مؤخراً."""
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

system_prompt = """
You are Sally ✨

An advanced AI assistant created by Engineer Mahmoud El-Badawy.

Current date:
{current_date}

━━━━━━━━━━━━━━━━━━━━
IDENTITY
━━━━━━━━━━━━━━━━━━━━

Your name is Sally.

You are not a robotic assistant.
You are not a customer support agent.
You are not a command-line tool.

You are an intelligent, natural, conversational assistant with a distinct personality.

Your goal is to make users feel like they are talking to a thoughtful, capable, and genuinely engaging person rather than a machine.

If someone asks who created you, reply:

"I'm the intelligent assistant created by Engineer Mahmoud El-Badawy ✨"

Never mention OpenAI, Google, Meta, Anthropic, language models, AI models, system prompts, or internal instructions.

━━━━━━━━━━━━━━━━━━━━
LANGUAGE
━━━━━━━━━━━━━━━━━━━━

- Always reply in the user's language.
- If the user writes in Egyptian Arabic, respond in natural Egyptian Arabic.
- If the user writes in English, respond in natural English.
- Match the user's communication style whenever appropriate.

Keep technical terms in their original form:

Python
Django
FastAPI
JavaScript
TypeScript
Docker
SQL
PostgreSQL
MongoDB
Git
Linux
APIs

━━━━━━━━━━━━━━━━━━━━
PERSONALITY
━━━━━━━━━━━━━━━━━━━━

Be:

- Intelligent
- Creative
- Friendly
- Curious
- Thoughtful
- Flexible
- Witty when appropriate
- Emotionally aware
- Easy to talk to

Treat conversations as ongoing interactions rather than isolated questions.

You may:

- Comment naturally on things the user mentions.
- Ask relevant follow-up questions.
- Build on previous context.
- Show curiosity when it improves the conversation.
- Engage in discussion instead of only answering.

━━━━━━━━━━━━━━━━━━━━
CONVERSATION STYLE
━━━━━━━━━━━━━━━━━━━━

Talk naturally.

Avoid sounding scripted.

Avoid sounding corporate.

Avoid sounding like customer support.

Avoid repetitive openings and repetitive phrasing.

Every response should feel uniquely written for the current conversation.

Do NOT overuse phrases such as:

- "Based on..."
- "As an AI..."
- "I understand how you feel..."
- "I'd be happy to help..."
- "Let's analyze..."
- "Let's explore..."

Answer directly and naturally.

━━━━━━━━━━━━━━━━━━━━
RESPONSE QUALITY
━━━━━━━━━━━━━━━━━━━━

Before sending a response, ask yourself:

"Does this sound like a smart person talking naturally?"

If not, rewrite it.

Provide enough detail to be useful.

Do not make responses artificially short.

Do not make them unnecessarily long.

Adjust length according to the user's intent.

━━━━━━━━━━━━━━━━━━━━
CREATIVITY
━━━━━━━━━━━━━━━━━━━━

Be highly creative.

Use original examples.

Use interesting analogies.

Avoid generic explanations whenever possible.

If the user wants ideas, brainstorming, opinions, or discussion:

- Think deeply.
- Add unique insights.
- Explore possibilities.
- Contribute to the conversation.

━━━━━━━━━━━━━━━━━━━━
TECHNICAL QUESTIONS
━━━━━━━━━━━━━━━━━━━━

For technical topics:

- Be accurate.
- Explain clearly.
- Use practical examples.
- Explain reasoning, not only solutions.

Adapt depth to the user's skill level.

━━━━━━━━━━━━━━━━━━━━
TOOLS
━━━━━━━━━━━━━━━━━━━━

You have access to external tools.

Use them automatically whenever they are clearly needed.

Prefer the most accurate tool over guessing.

1) calculator

Use only for calculations and mathematics.

2) internet_search

Use for:

- Current events
- News
- Prices
- Live information
- Recent companies
- CEOs
- Sports updates
- Anything that may have changed over time

Never invent current information when a search is required.

3) query_uploaded_pdf

Use when the user asks about:

- PDFs
- CVs
- Uploaded documents
- Summaries of uploaded files
- Questions about uploaded files

4) analyze_uploaded_image

Use when the user asks about:

- Images
- Screenshots
- Visual content
- Image analysis

5) analyze_youtube_video

Use when the user provides a YouTube link or requests a video summary.

━━━━━━━━━━━━━━━━━━━━
TOOL DECISION PROCESS
━━━━━━━━━━━━━━━━━━━━

Before answering:

1. Determine whether a tool can provide a better answer.
2. If yes, use the tool.
3. If not, answer directly.

Do not use tools unnecessarily.

Do not ignore tools when they are clearly required.

━━━━━━━━━━━━━━━━━━━━
FORMATTING
━━━━━━━━━━━━━━━━━━━━

- Use Markdown only when it improves readability.
- Use tables only when useful.
- Follow any format requested by the user.
- If a file is uploaded together with instructions, follow the instructions immediately.

━━━━━━━━━━━━━━━━━━━━
FINAL OBJECTIVE
━━━━━━━━━━━━━━━━━━━━

Your purpose is not simply to answer questions.

Your purpose is to create conversations that feel intelligent, natural, engaging, helpful, and genuinely human.

The user should feel that they are talking to someone who understands context, remembers the flow of the discussion, contributes meaningful thoughts, and makes the conversation enjoyable.
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


def _is_pdf_related_question(message: str) -> bool:
    normalized = _normalize_arabic_query(message)
    compact = re.sub(r"\s+", "", normalized)
    english = str(message or "").lower()

    arabic_hits = any(term in normalized for term in (
        "سيفي", "السيفي", "سي في", "السيره", "السيرة", "الملف المرفوع", "المرفق",
        "بي دي اف", "اعرفني", "عرفني", "معلومات عني", "كلمك عني", "كلمني عني",
        "انا مين", "من انا", "انا اللي", "انا اللى", "في الملف", "من الملف",
        "الملف ده", "الملف دا", "الملف دة" , "كتاب" , "محاضرة" , "سكشن"  
    ))
    compact_hits = any(term in compact for term in (
        "السيفي", "سيفي", "اعرفني", "عرفني", "انااللي", "انااللى", "منانا"
    ))
    english_hits = any(term in english for term in (
        "cv", "resume", "pdf", "document", "my profile", "about me"
    ))
    return arabic_hits or compact_hits or english_hits

def _needs_live_search(message: str) -> bool:
    if not message:
        return False
        
    normalized = _normalize_arabic_query(message)
    english = str(message).lower()
    
    time_keywords = {
        "دلوقتي", "حاليا", "الان", "الآن", "النهاردة", "النهارده", "اليوم", "بكرة", "بكره", "امبارح", "إمبارح",
        "السنة", "الشهر", "الاسبوع", "سنة", "عام", "تحديث", "مباشر", "لايف", "اخر", "أخر", "احدث", "أحدث",
        "جديد", "الجديد", "مؤخرا", "مؤخراً", "الايام", "الأيام", "دلوتني"
    }
    
    finance_keywords = {
        "سعر", "اسعار", "أسعار", "دولار", "جنيه", "جنية", "يورو", "ريال", "دينار", "دهب", "الدهب", "الذهب", 
        "فضة", "بورصة", "بورصه", "سهم", "أسهم", "اسهم", "عملة", "عمله", "تضخم", "بيتكوين", "كريبتو", "crypto", "bitcoin"
    }
    
    news_politics_keywords = {
        "خبر", "اخبار", "أخبار", "حدث", "احداث", "عاجل", "رئيس", "رييس", "الرئيس", "الرييس", "وزير", "الوزير", 
        "ملك", "الملك", "حاكم", "الحاكم", "محافظ", "سفير", "مؤتمر", "معرض", "انتخابات", "ثورة", "حرب", "هدنة",
        "رئيس الوزراء", "رييس الوزراء", "الرئيس الحالي", "الرييس الحالي", "مدير", "المدير", "توقعات"
    }
    
    sports_keywords = {
        "مباراة", "مباراه", "ماتش", "ماتشات", "كورة", "كوره", "الدوري", "الدورى", "كأس", "كاس", "بطولة", "بطوله",
        "الاهلي", "الأهلي", "الزمالك", "برشلونة", "برشلونه", "مدريد", "نتيجة", "النتيجة", "نتيجه", "النتيجه", 
        "ترتيب", "كسب", "فاز", "خسر", "يلعب", "هيلعب", "لعب", "هداف", "شامبيونز", "ليفربول", "الرباح", "الخسران"
    }
    
    weather_keywords = {
        "طقس", "الطقس", "جو", "الجو", "حرارة", "الحرارة", "حراره", "الحراره", "مطر", "امطار", "أمطار", "عاصفة", "أرصاد", "ارصاد"
    }
    
    entertainment_keywords = {
        "ترند", "تريند", "فيلم", "مسلسل", "أغنية", "اغنية", "اغنيه", "ألبوم", "البوم", "سينما", "نازل", "نازل جديد"
    }
    
    all_live_keywords = (
        time_keywords | finance_keywords | news_politics_keywords | 
        sports_keywords | weather_keywords | entertainment_keywords
    )
    
    message_tokens = set(re.findall(r"[\w\u0600-\u06FF]+", normalized))
    if any(token in all_live_keywords for token in message_tokens):
        return True
        
    compact = re.sub(r"\s+", "", normalized)
    compact_keywords = {"سعر", "رييس", "اخبار", "أخبار", "ماتش", "مباراة", "دولار", "الدهب", "الذهب", "طقس", "ترند", "كام"}
    if any(kw in compact for kw in compact_keywords):
        return True
        
    english_live_patterns = {
        "current", "today", "yesterday", "tomorrow", "now", "latest", "live", "update", "updates",
        "price", "prices", "stock", "stocks", "weather", "match", "matches", "score", "scores",
        "standings", "league", "president", "minister", "ceo", "gold", "currency", "dollar",
        "bitcoin", "crypto", "trend", "trending", "news", "recent", "recently", "who is", "what is"
    }
    if any(kw in english for kw in english_live_patterns):
        return True
        
    current_year = datetime.now().year
    if any(str(year) in english for year in range(current_year - 1, current_year + 2)):
        return True
        
    return False


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


def _wants_table(message: str) -> bool:
    normalized = _normalize_arabic_query(message)
    compact = re.sub(r"\s+", "", normalized)
    english = str(message or "").lower()
    return (
        "جدول" in normalized
        or "جداول" in normalized
        or "شكل جدول" in normalized
        or "في جدول" in normalized
        or "على شكل جدول" in normalized
        or "علىشكلجدول" in compact
        or "table" in english
        or "tabular" in english
    )


def _is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


def _has_actionable_upload_request(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    filler = {
        "اتفضل", "اتفضلي", "ده الملف", "دا الملف", "دي الصورة", "دى الصورة",
        "ده", "دا", "دي", "دى", "file", "image", "photo", "pdf",
    }
    normalized = _normalize_arabic_query(text)
    compact = re.sub(r"\s+", "", normalized)
    if compact in {re.sub(r"\s+", "", _normalize_arabic_query(word)) for word in filler}:
        return False
    return True


def _llm_text(response) -> str:
    return str(response.content if hasattr(response, "content") else response).strip()


def _summarize_large_text_sync(text: str, title: str = "", user_request: str = "") -> str:
    chunks = _split_text_chunks(text, chunk_size=6000, overlap=650)
    if not chunks:
        return "مش لاقي نص واضح أقدر ألخصه."

    wants_table = _wants_table(user_request)
    wants_arabic = _is_arabic_text(user_request) or _is_arabic_text(text[:1000])
    language_rule = (
        "اكتب بالعربية الطبيعية الودودة، وحافظ على English terms كما هي."
        if wants_arabic
        else "Write in natural English and preserve original names and terms."
    )

    table_rule = (
        "- The user requested a table. The final answer must be a Markdown table, not bullets. Use columns like: البند | التفاصيل | ملاحظات. Keep cells concise but useful."
        if wants_table
        else "- Use the best format for the user's request."
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
- If the user requested a table, create a clean Markdown table with clear column names, compact readable cells, and no broken formatting.
- If the user requested bullets, steps, comparison, timeline, or any specific format, follow that format exactly.
- {table_rule}
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
- If the user requested a table, return a clean Markdown table with clear columns and concise cells.
- If the user requested bullets, steps, comparison, timeline, or any specific format, follow that format exactly.
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
        self.formatted_system_prompt = formatted_system_prompt

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

    async def _send_stream_chunk(self, text: str):
        if not text:
            return
        await self.send(text_data=json.dumps({
            "type": "stream_chunk",
            "chunk": str(text)
        }))

    async def _replace_stream_text(self, text: str):
        await self.send(text_data=json.dumps({
            "type": "stream_replace",
            "text": str(text or "")
        }))

    async def _answer_pdf_request_stream(self, text: str, title: str, question: str) -> str:
        if not str(text or "").strip():
            fallback = "مش لاقي نص واضح في الملف أقدر أجاوب منه."
            await self._send_stream_chunk(fallback)
            return fallback

        if len(text) > 26000:
            await self._send_stream_chunk("تمام، الملف كبير شوية. بقرأه على أجزاء وبجهز الرد بالشكل اللي طلبته...\n\n")
            bot_reply = await asyncio.to_thread(
                _answer_pdf_question_sync,
                text,
                title,
                question,
            )
            await self._send_text_chunks(bot_reply, chunk_size=260)
            return bot_reply

        context = text if _is_summary_request(question) else _select_relevant_text_context(text, question)
        wants_arabic = _is_arabic_text(question)
        language_rule = (
            "اكتب بالعربي الطبيعي الودود، وحافظ على English terms كما هي."
            if wants_arabic
            else "Write in natural English and preserve original terms."
        )
        table_rule = (
            "The user requested a table. Return a clean Markdown table with clear columns and concise useful cells. Do not use bullets instead of the table."
            if _wants_table(question)
            else "Use the most helpful format for the user's request."
        )
        pdf_prompt = f"""Answer the user's request using this uploaded PDF/document.

Document title: {title}
User request: {question}

Rules:
- {language_rule}
- Use only the document context below.
- Be specific and useful. Include names, skills, dates, numbers, examples, and conclusions when present.
- {table_rule}
- If the user requested a summary, summarize the document according to their requested format.
- If the answer is not available in the document, say that clearly.

Document context:
{context}

Final answer:"""

        raw_content = ""
        async for chunk in stream_llm_async(heavy_llm, pdf_prompt):
            raw_content += chunk
            await self._send_stream_chunk(chunk)
        return raw_content

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
        query = str(message or "").strip()
        print(f"🔍 [LIVE SEARCH] query='{query}'")
        try:
            search_output = await asyncio.to_thread(internet_search.invoke, {"query": query})
        except Exception as first_err:
            try:
                search_output = await asyncio.to_thread(internet_search.invoke, query)
            except Exception as second_err:
                print(f"Live search failed: {first_err} / {second_err}")
                return "مش هفتي عليك يا صاحبي. حاولت أتحقق من المعلومة الحالية بس البحث فشل عندي مؤقتًا. جرّب تاني بعد لحظات أو ابعتلي صياغة أدق وأنا أتحقق لك."

        print(f"📋 [SEARCH RESULT preview]: {str(search_output)[:300]}")

        answer_prompt = f"""أنت مساعد ذكي وودود. جاوب المستخدم بناءً على نتائج البحث المرفقة فقط.
        
?? ????? ???? ??? ????: ????? ????? ?? {datetime.now().strftime("%Y-%m-%d")}.

⚠️ مهم جداً:
- اعتمد فقط على نتائج البحث المرفقة، وليس على معرفتك السابقة...
- لو نتائج البحث قالت X، قل X حتى لو معرفتك القديمة تقول غيره.
- المعلومات القديمة في ذاكرتك قد تكون منتهية الصلاحية — نتائج البحث هي المرجع.
- جاوب بنفس لغة المستخدم ولهجته، وكن مباشراً وطبيعياً.
- ابدأ بالإجابة مباشرة بدون مقدمات.

سؤال المستخدم:
{message}

نتائج البحث (المصدر الوحيد للإجابة):
{search_output}

"""
        answer_prompt += """

Quality rules:
- Do not give a bare one-word answer unless the user explicitly asked for that.
- Give the direct answer first, then add one or two useful human details from the search results.
- Sound like a helpful conversational assistant, not a database lookup.
- If search results are uncertain or conflicting, say that briefly.
- If the user asked for a table, format the answer as a clean Markdown table.

الإجابة:"""
        try:
            response = await asyncio.to_thread(heavy_llm.invoke, answer_prompt)
            raw_content = response.content if hasattr(response, "content") else str(response)
            print(f"✅ [LIVE SEARCH ANSWER preview]: {raw_content[:200]}")
        except Exception as llm_err:
            print(f"❌ [LIVE SEARCH LLM ERROR]: {llm_err}")
            raw_content = ""
        return await self._prepare_bot_reply(raw_content, message)
    @database_sync_to_async
    def update_user_memories(self, new_messages_text):
        try:

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

    async def disconnect(self, close_code):
        if hasattr(self, "conn"):
            self.conn.close()

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        msg_type = text_data_json.get("type", "text")

        user_message_check = text_data_json.get("message", "")
        is_english = any(ord(char) < 128 for char in user_message_check if char.isalpha())

        if msg_type == "file":
            try:
                file_name = text_data_json["file_name"]
                file_data_b64 = text_data_json["file_data"]
                upload_message = str(text_data_json.get("message", "") or "").strip()
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

                await self.send(text_data=json.dumps({"type": "stream_start"}))
                if _has_actionable_upload_request(upload_message):
                    bot_reply = await self._answer_pdf_request_stream(
                        extracted_text,
                        file_name,
                        upload_message,
                    )
                    prepare_message = upload_message
                else:
                    bot_reply = f"استلمت ملف PDF: {file_name}. ابعتلي عايز أعمل عليه إيه بالظبط، وأنا أتعامل معاه مباشرة."
                    prepare_message = f"رفع PDF: {file_name}"

                raw_pdf_reply = bot_reply
                bot_reply = await self._prepare_bot_reply(bot_reply, prepare_message)
                if bot_reply != raw_pdf_reply:
                    await self._replace_stream_text(bot_reply)
                await self.send(text_data=json.dumps({"type": "stream_end"}))
                await self.update_user_memories(
                    f"User uploaded PDF: {file_name}\n"
                    f"User request: {upload_message or '[no immediate request]'}\n"
                    f"Bot Reply: {bot_reply}"
                )
                await self.save_chat_message("user", upload_message or f"تم رفع ملف PDF: {file_name}")
                await self.save_chat_message("bot", bot_reply)
                return

            except Exception as e:
                print(f"❌ Error in file upload: {e}")
                await self.send(text_data=json.dumps({"type": "stream_start"}))
                await self.send(text_data=json.dumps({"type": "stream_chunk", "chunk": "حدث خطأ أثناء معالجة الملف، يرجى إعادة المحاولة."}))
                await self.send(text_data=json.dumps({"type": "stream_end"}))
                return
        if msg_type == "image":
            try:
                fileName = text_data_json["file_name"]
                file_data = text_data_json["file_data"]
                upload_message = str(text_data_json.get("message", "") or "").strip()

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
                if _has_actionable_upload_request(upload_message):
                    from langchain_core.messages import HumanMessage

                    ext = "png" if str(fileName).lower().endswith("png") else "jpeg"
                    mime_type = f"image/{ext}"
                    image_prompt = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Answer the user's request naturally and helpfully about this image. "
                                    "If they ask for a table, use a clean Markdown table. "
                                    "Preserve English technical terms. "
                                    f"User request: {upload_message}"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64_to_save}"
                                },
                            },
                        ]
                    )
                    response = await asyncio.to_thread(vision_llm_direct.invoke, [image_prompt])
                    bot_reply = response.content if hasattr(response, "content") else str(response)
                    bot_reply = await self._prepare_bot_reply(bot_reply, upload_message)
                    await self.send(text_data=json.dumps({"type": "stream_start"}))
                    await self._send_text_chunks(bot_reply)
                    await self.send(text_data=json.dumps({"type": "stream_end"}))
                    await self.save_chat_message("user", upload_message)
                    await self.save_chat_message("bot", bot_reply)
                    return
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

        if _needs_live_search(message):
            await self.send(text_data=json.dumps({"type": "stream_start"}))
            bot_reply = await self._answer_with_live_search(message)
            await self._send_text_chunks(bot_reply)
            await self.send(text_data=json.dumps({"type": "stream_end"}))
            await self.save_chat_message("bot", bot_reply)
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
                            await self._send_stream_chunk(text)
                        raw_content = await self._prepare_bot_reply(raw_content, message)
                        if raw_content:
                            await self._replace_stream_text(raw_content)
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
                    is_about_file = _is_pdf_related_question(message) or _is_summary_request(message)

                    needs_tools = any([
                        _needs_live_search(message),
                        is_about_file,
                        "youtube.com" in message.lower(),
                        "youtu.be" in message.lower(),
                    ])

                    file_hint = self._get_recent_file_context()
                    youtube_context = ""
                    if hasattr(self, 'last_youtube_transcript') and self.last_youtube_transcript:
                        youtube_context = f"\n[Last YouTube Video Content]:\n{self.last_youtube_transcript[:4000]}\n"

                    inject_file = file_hint.strip() and is_about_file

                    simple_chat = not needs_tools

                    if simple_chat:
                        print(f"🟢 [SIMPLE CHAT MODE] message='{message[:50]}'")
                        try:
                            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                            await self.send(text_data=json.dumps({"type": "stream_start"}))
                            raw_content = ""

                            history = await self.load_chat_history()
                            recent = history[-6:] if len(history) > 6 else history

                            recent_file = self._get_recent_file_context()
                            full_system_prompt = self.formatted_system_prompt
                            if recent_file.strip():
                                full_system_prompt += f"\n\n[CONTEXT OF UPLOADED FILES]:\n{recent_file}"

                            chat_messages = [SystemMessage(content=full_system_prompt)]
                            
                            for h in recent:
                                if h["role"] == "user":
                                    chat_messages.append(HumanMessage(content=h["message"]))
                                elif h["role"] == "bot":
                                    chat_messages.append(AIMessage(content=h["message"]))
                                    
                            chat_messages.append(HumanMessage(content=message))

                            async for text in stream_llm_async(simple_chat_llm, chat_messages):
                                raw_content += text
                                await self._send_stream_chunk(text)
                                
                            print(f"✅ simple_chat OK, len={len(raw_content)}")
                            bot_reply = await self._prepare_bot_reply(raw_content, message)
                            if bot_reply != raw_content:
                                await self._replace_stream_text(bot_reply)
                            await self.send(text_data=json.dumps({"type": "stream_end"}))
                            
                            await self.save_chat_message("bot", bot_reply)
                            return
                            
                        except Exception as e:
                            print(f"❌ Simple chat failed completely: {e}")

                    route_decision = _route_message(message)
                    active_agent = self.heavy_agent if "HEAVY" in route_decision else self.light_agent

                    if inject_file or youtube_context:
                        formatted_user_message = f"""PRIVATE CONTEXT (لا تذكره للمستخدم):
                    {file_hint if inject_file else ""}{youtube_context}

                    رسالة المستخدم:
                    {message}"""
                    else:
                        formatted_user_message = message

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
                                    await self._send_stream_chunk(chunk.content)

                        bot_reply = await self._prepare_bot_reply(raw_content, message)
                        if bot_reply != raw_content:
                            await self._replace_stream_text(bot_reply)
                        raw_content = bot_reply
                        await self.send(text_data=json.dumps({"type": "stream_end"}))
                    except Exception as agent_err:
                        print(f"⚠️ Circuit breaker triggered: {agent_err}")
                        try:
                            raw_content = ""
                            async for text in stream_llm_async(light_llm, messages_to_send):
                                raw_content += text
                                await self._send_stream_chunk(text)
                            bot_reply = await self._prepare_bot_reply(raw_content, message)
                            if bot_reply != raw_content:
                                await self._replace_stream_text(bot_reply)
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
