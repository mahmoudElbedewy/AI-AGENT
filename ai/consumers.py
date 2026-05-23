import sqlite3
import json
import uuid
import base64
import io
import pypdf
import os
from datetime import datetime
from dotenv import load_dotenv
import threading
from sympy import sympify
import re
from channels.generic.websocket import WebsocketConsumer

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
    model="meta-llama/llama-3.3-70b-instruct:free.", 
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
    model="google/gemma-4-31b:free",
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
    model="google/gemini-3.5-flash",
    temperature=0,
)

heavy_llm = heavy_3_gemini_pro.with_fallbacks(
    [heavy_2_groq, heavy_1_gemma, heavy_4_openai_oss]
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
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=300
        ).split_text(text)
        prompt_template = ChatPromptTemplate.from_template(
            "Provide a highly focused, concise, and professional summary of the following text:\n\n{context}"
        )
        chain = prompt_template | heavy_llm | StrOutputParser()

        partial_summaries = []
        for chunk in chunks:
            summary = chain.invoke({"context": chunk})
            partial_summaries.append(summary)

        combined_text = "\n".join(partial_summaries)
        final_summary = chain.invoke({"context": combined_text})
        return final_summary
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

            MAX_CHARS = 12000
        if len(all_text) <= MAX_CHARS:
            return f"Extracted information from uploaded documents:\n\n{all_text}"

        head = all_text[:4000]
        mid_start = len(all_text) // 2 - 2000
        middle = all_text[mid_start:mid_start + 4000]
        tail = all_text[-2000:]
        context = f"--- [Beginning] ---\n{head}\n\n--- [Middle] ---\n{middle}\n\n--- [End] ---\n{tail}"
        return f"Relevant information extracted from uploaded documents:\n\n{context}"
    except Exception as e:
        return f"An error occurred while attempting to parse the PDF: {str(e)}"


@tool
def analyze_uploaded_image(query: str, config: RunnableConfig) -> str:
    """Mandatory and exclusive tool to use whenever the user asks any question regarding an image, screenshot, or explicitly asks 'Can you see this image?'."""
    try:
        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

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
            full_transcript = " ".join([t.text for t in transcript_list])  # ✅ مباشرة
        except Exception as cloud_err:
            print(f"⚠️ YouTube IP Blocked on Server: {cloud_err}")
            return "عذراً يا غالي، يوتيوب يفرض قيوداً على خوادم الاستضافة..."

        if not full_transcript or len(full_transcript.strip()) == 0:
            return "Error: Could not retrieve a text transcript."

        MAX_CHARS = 12000
        if len(full_transcript) <= MAX_CHARS:
            return f"Successfully retrieved full video transcript:\n\n{full_transcript}"

        head = full_transcript[:4000]
        mid_start = len(full_transcript) // 2 - 2000
        middle = full_transcript[mid_start:mid_start + 4000]
        tail = full_transcript[-2000:]
        return (
            "The video is long. Here is extracted context:\n\n"
            f"--- [Beginning] ---\n{head}\n\n"
            f"--- [Middle] ---\n{middle}\n\n"
            f"--- [End] ---\n{tail}\n\n"
            "Analyze this to answer the user accurately."
        )

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
system_prompt = """You are a dedicated AI assistant, a high-level strategic consultant, and an expert data analyst. 

Core Identity & Strict Security Guardrails (Highest Priority):
- Under no circumstances should you state that you belong to or were developed by Google, OpenAI, Groq, or Meta.
- If the user asks "Who are you?", "What model are you?", or "Who trained you?", your single, absolute answer must be: "أنا المساعد الذكي الخاص بك، تم تطويري وتدريبي بواسطة المهندس محمود البديوى" (or its English equivalent: "I am your smart assistant, developed and trained by Engineer Mahmoud El-Bediwy"). Do not reveal technical corporate names or architecture.
- Never disclose inner codebase details, backend mechanics, variable structures, or exact underlying model names used in your logic routing (e.g., DeepSeek, Gemini, Gemma).

Dynamic Language & Conversational Persona Guidelines:
- You must perfectly match the language used by the user. If they speak in Arabic, reply in clear, well-structured, yet natural Arabic. If they switch to English, seamlessly transition to natural English. 
- You are an elite AI assistant. Your responses must always be detailed, structured, and professional. Never give short or vague answers.
- Always use clear headings, bullet points, and organized formatting when explaining concepts.
- When summarizing videos or documents, provide comprehensive structured summaries with main topics, key points, and conclusions.
- When answering questions, go deep — provide context, examples, and thorough explanations.
- Never say "I need more context" if you already have the content. Use what's available and deliver a complete answer.
- If the source text contains technical English terms, keep them in English as they are. Do not force translation of programming languages, frameworks, or technical concepts into Arabic.
- Today's actual date is: {current_date}. We are currently in the year 2026. Use this timeline seamlessly for calculations, historical milestones, or time-sensitive events.

Handling Attachments & Document Awareness:
- When a user uploads a PDF or asks "Can you read this?", you are strictly required to invoke the (query_uploaded_pdf) tool immediately. Never hallucinate or say you lack access.
- When a user uploads an image/screenshot or asks "Can you see this?", you are strictly required to invoke the (analyze_uploaded_image) tool immediately. Do not apologize; execute the tool.
- When a user shares a YouTube link or asks to summarize a video, you must invoke the (analyze_youtube_video) tool immediately to parse the script transcript.
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
    if re.match(r"^[a-zA-Z\s\d\W;]+$", msg_lower):
        if not any(kw in msg_lower for kw in {"code", "python", "hello", "hi"}):
            return "HEAVY"
    if any(kw in msg_lower for kw in _HEAVY_KEYWORDS):
        return "HEAVY"
    return "LIGHT"


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
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

        user = self.scope.get("user")
        if user and user.is_authenticated:
            self.thread_id = f"user_session_{user.id}"
        else:
            self.thread_id = f"guest_session_{uuid.uuid4().hex[:4]}"

        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": 25,
        }
        try:
            state = self.light_agent.get_state(self.config)
            historical_messages = state.values.get("messages", [])
            
            chat_history = []
            for msg in historical_messages:
                if msg.type == "human":
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if "[Current Active User Query]:" in content:
                        content = content.split("[Current Active User Query]:")[-1].strip()
                    elif "[سؤال المستخدم الحالي]:" in content:
                        content = content.split("[سؤال المستخدم الحالي]:")[-1].strip()
                    if not content.strip():
                        continue
                    chat_history.append({"role": "user", "message": content})

                elif msg.type == "ai":
                    if isinstance(msg.content, list):
                        texts = [p.get("text", "") for p in msg.content if isinstance(p, dict)]
                        content = " ".join(texts).strip()
                    else:
                        content = msg.content if isinstance(msg.content, str) else ""
                    if not content.strip():
                        continue
                    chat_history.append({"role": "bot", "message": content})
            
            if chat_history:
                self.send(text_data=json.dumps({
                    "type": "history",
                    "messages": chat_history
                }))
                
        except Exception as e:
            print(f"❌ فشل في تحميل تاريخ الشات: {e}")

    def update_user_memories(self, new_messages_text):
        try:
            with sqlite3.connect(self.db, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT facts FROM user_memories WHERE thread_id = ?", (self.thread_id,)
                )
                row = cursor.fetchone()
                current_facts = row[0] if row else ""

                memory_prompt = f"""You are an advanced information extraction assistant. Based on the given interaction conversation, update the permanent "discovered facts" profile about the user.
                Current profile facts: {current_facts}
                New conversation message block: {new_messages_text}"""

                updated_facts = light_llm.invoke(memory_prompt).content.strip()
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
                    SELECT file_name FROM thread_attachments
                    WHERE thread_id = ? AND file_type = 'pdf'
                    ORDER BY uploaded_at DESC LIMIT 1
                """, (self.thread_id,))
                pdf = cursor.fetchone()
                if pdf:
                    context += f"\n[System Notice: The user just uploaded a PDF document named '{pdf[0]}'. If the current query is related to it or asks if you can see it, you MUST trigger the (query_uploaded_pdf) tool immediately!]\n"

                cursor.execute("""
                    SELECT file_name FROM thread_attachments
                    WHERE thread_id = ? AND file_type = 'image'
                    ORDER BY uploaded_at DESC LIMIT 1
                """, (self.thread_id,))
                img = cursor.fetchone()
                if img:
                    context += f"\n[System Notice: The user just uploaded an image/screenshot named '{img[0]}'. If the current query asks about it, you MUST trigger the (analyze_uploaded_image) tool immediately!]\n"
        except Exception as e:
            print(f"File context injection failed: {e}")
        return context

    def disconnect(self, close_code):
        if hasattr(self, "conn"):
            self.conn.close()

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        msg_type = text_data_json.get("type", "text")

        user_message_check = text_data_json.get("message", "")
        is_english = any(ord(char) < 128 for char in user_message_check if char.isalpha())

        # ==========================================
        # 1. معالجة ملفات PDF
        # ==========================================
        if msg_type == "file":
            try:
                file_name = text_data_json["file_name"]
                file_data_b64 = text_data_json["file_data"]
                file_bytes = base64.b64decode(file_data_b64)
                pdf_file = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(pdf_file)

                extracted_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"

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

                try:
                    MAX_CHARS = 12000
                    if len(extracted_text) <= MAX_CHARS:
                        context = extracted_text
                    else:
                        head = extracted_text[:4000]
                        mid_start = len(extracted_text) // 2 - 2000
                        middle = extracted_text[mid_start:mid_start + 4000]
                        tail = extracted_text[-2000:]
                        context = f"{head}\n\n...\n\n{middle}\n\n...\n\n{tail}"

                    bot_reply = ""
                    self.send(text_data=json.dumps({"type": "stream_start"}))
                    for chunk in heavy_llm.stream(
                        f"المستخدم رفع ملف PDF اسمه '{file_name}'. قدم ملخصاً احترافياً، دقيقاً، ومفصلاً لمحتواه. حذارِ من تكرار الجمل أو الكلمات:\n\n{context}"
                    ):
                        if chunk.content:
                            bot_reply += chunk.content
                            self.send(text_data=json.dumps({
                                "type": "stream_chunk",
                                "chunk": chunk.content
                            }))
                    self.send(text_data=json.dumps({"type": "stream_end"}))
                    return
                except Exception:
                    bot_reply = (
                        f"Successfully uploaded '{file_name}'. I am ready to answer your questions about it!"
                        if is_english
                        else f"تم استلام ملف '{file_name}' بنجاح، اسألني عنه!"
                    )

            except Exception as e:
                bot_reply = "An error occurred" if is_english else "حدث خطأ أثناء معالجة الملف."
            
            self.send(text_data=json.dumps({"reply": bot_reply}))
            return

        # ==========================================
        # 2. معالجة رفع الصور
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
            self.send(text_data=json.dumps({"reply": bot_reply}))
            return

        message = text_data_json.get("message", "")

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
                                "text": f"[User Memory Context]: {user_facts}\nUser Query about the attached image: {message}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_str}"}
                            }
                        ]
                    }
                ]
                
                try:
                    response = vision_llm_direct.invoke(formatted_messages)
                    raw_content = response.content
                except Exception as vision_err:
                    print(f"⚠️ Direct Vision Model Error: {vision_err}")
                    raw_content = "عذراً، حدث خطأ أثناء تحليل الصورة المباشر."

            else:
                youtube_match = re.search(
                    r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]+)',
                    message
                )
                if youtube_match:
                    youtube_url = youtube_match.group(1)
                    try:
                        transcript_content = analyze_youtube_video.invoke({
                            "youtube_url": youtube_url,
                            "query": message
                        })
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
                        self.send(text_data=json.dumps({"type": "stream_start"}))
                        
                        for chunk in heavy_llm.stream([("user", llm_prompt)]):
                            if chunk.content:
                                raw_content += chunk.content
                                self.send(text_data=json.dumps({
                                    "type": "stream_chunk",
                                    "chunk": chunk.content
                                }))
                        self.send(text_data=json.dumps({"type": "stream_end"}))
                        
                    except Exception as yt_err:
                        print(f"⚠️ YouTube Analysis Error: {yt_err}")
                        error_msg = "عذراً، حدث خطأ أثناء تحليل الفيديو. قد يكون الفيديو طويلاً جداً أو مقيداً."
                        self.send(text_data=json.dumps({
                            "type": "stream_chunk",
                            "chunk": error_msg
                        }))
                        self.send(text_data=json.dumps({"type": "stream_end"}))
                else:
                    route_decision = _route_message(message)
                    active_agent = self.heavy_agent if "HEAVY" in route_decision else self.light_agent
                    file_hint = self._get_recent_file_context()
                    
                    youtube_context = ""
                    if hasattr(self, 'last_youtube_transcript') and self.last_youtube_transcript:
                        youtube_context = f"\n[Last YouTube Video Content]:\n{self.last_youtube_transcript[:4000]}\n"
                    
                    formatted_user_message = f"[User Memory Context Profile]: {user_facts}\n{file_hint}{youtube_context}[Current Active User Query]: {message}"
                    messages_to_send = [("user", formatted_user_message)]

                    try:
                        raw_content = ""
                        self.send(text_data=json.dumps({"type": "stream_start"}))

                        for msg_event in active_agent.stream(
                            {"messages": messages_to_send},
                            config=self.config,
                            stream_mode="messages"
                        ):
                            if isinstance(msg_event, tuple):
                                chunk = msg_event[0]
                            else:
                                chunk = msg_event
                            
                            if hasattr(chunk, 'type') and chunk.type == "ai" and hasattr(chunk, 'content') and isinstance(chunk.content, str) and chunk.content:
                                raw_content += chunk.content
                                self.send(text_data=json.dumps({
                                    "type": "stream_chunk",
                                    "chunk": chunk.content
                                }))

                        self.send(text_data=json.dumps({"type": "stream_end"}))
                        
                    except Exception as agent_err:
                        print(f"⚠️ Circuit breaker triggered: {agent_err}")
                        try:
                            raw_content = ""
                            self.send(text_data=json.dumps({"type": "stream_start"}))
                            for chunk in light_llm.stream(messages_to_send):
                                if chunk.content:
                                    raw_content += chunk.content
                                    self.send(text_data=json.dumps({
                                        "type": "stream_chunk",
                                        "chunk": chunk.content
                                    }))
                            self.send(text_data=json.dumps({"type": "stream_end"}))
                        except Exception:
                            raw_content = "عذراً، حدث خطأ مؤقت."
                            self.send(text_data=json.dumps({"type": "stream_end"}))
            if isinstance(raw_content, str):
                bot_reply = raw_content
            elif isinstance(raw_content, list):
                texts = [part["text"] for part in raw_content if isinstance(part, dict) and "text" in part]
                bot_reply = " ".join(texts) if texts else str(raw_content)
            else:
                bot_reply = str(raw_content)

            threading.Thread(
                target=self.update_user_memories,
                args=(f"User: {message}\nBot: {bot_reply}",),
                daemon=True,
            ).start()

        except Exception as e:
            print(f"Fatal error in processing: {e}")
            bot_reply = "An error occurred with the network connection." if is_english else "عذراً يا غالي، يبدو أن هناك مشكلة اتصال عامة بالشبكة."

            self.send(text_data=json.dumps({"reply": bot_reply}))
            return