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
from channels.generic.websocket import WebsocketConsumer

# LangChain
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ====================== LLMS ======================
light_1_deepseek = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-v4-flash:free",
    temperature=0,
)

light_2_gemini_direct = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_retries=1
)

light_3_groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY_1"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_retries=1
)

light_4_openai_oss = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-20b:free",
    temperature=0,
)

light_5_llama_or = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0,
)

light_llm = light_1_deepseek.with_fallbacks([
    light_2_gemini_direct, 
    light_3_groq, 
    light_4_openai_oss, 
    light_5_llama_or
])

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
    max_retries=1
)

heavy_3_gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=1.0, 
    max_retries=1
)

heavy_4_openai_oss = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free",
    temperature=0,
)

heavy_llm = heavy_1_gemma.with_fallbacks([
    heavy_2_groq, 
    heavy_3_gemini_pro, 
    heavy_4_openai_oss
])

# ====================== Tools ======================

@tool
def calculator(expression: str) -> str:
    """تستخدم كآلة حاسبة للعمليات الرياضية الحسابية فقط."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"خطأ في الحساب: {e}"

searchMethod = DuckDuckGoSearchRun()

def _run_search(query: str, container: list):
    try:
        container[0] = searchMethod.invoke(query)
    except Exception as e:
        container[0] = f"ERROR:{str(e)}"

@tool
def internet_search(query: str) -> str:
    """تستخدم للبحث في الإنترنت بجلب معلومات محدثة عن الشخصيات، الحكام، الأخبار، والأحداث الجارية."""
    container = [None]
    t = threading.Thread(target=_run_search, args=(query, container), daemon=True)
    t.start()
    t.join(timeout=7)

    result = container[0]
    if result is None:
        return "انتهت مهلة البحث. أجب من معلوماتك الحالية."
    if isinstance(result, str) and result.startswith("ERROR:"):
        return "فشل الاتصال بالإنترنت حالياً."
    if not result or len(result.strip()) < 20:
        return "لم تسفر نتائج البحث عن معلومات كافية."

    return f"نتائج البحث الجارية: {str(result)[:800]}"

@tool
def summarize_text_tool(text: str) -> str:
    """استخدم هذه الأداة وجوباً فقط عندما يطلب المستخدم تلخيص نص طويل، مقال، أو كتاب."""
    try:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=300
        ).split_text(text)
        prompt_template = ChatPromptTemplate.from_template(
            "قم بتلخيص النص التالي بأسلوب مركّز ومفيد:\n\n{context}"
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
        return f"فشل التلخيص: {str(e)}"

@tool
def query_uploaded_pdf(query: str, config: RunnableConfig) -> str:
    """استخدم هذه الأداة وجوباً وحصراً عندما يسألك المستخدم أي سؤال بخصوص ملفات الـ PDF التي قام برفعها، أو عندما يسألك 'شايف الملف دا؟' أو يطلب تلخيصه."""
    try:
        thread_id = config["configurable"].get("thread_id")
        print(f"🔍 [أداة الـ PDF]: جاري البحث عن ملفات للجلسة: {thread_id}")
        
        conn = sqlite3.connect("db.sqlite3")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_name, file_content FROM thread_attachments WHERE thread_id = ? AND file_type = 'pdf' ORDER BY uploaded_at DESC",
            (str(thread_id),),
        )
        rows = cursor.fetchall() 
        conn.close()
        
        if not rows:
            return "تنبيه للنظام: لم نجد ملفات PDF مرفوعة في قاعدة البيانات لهذه الجلسة حتى الآن."
        
        all_text = ""
        for row in rows:
            file_name = row[0]
            file_content = row[1] 
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8', errors='ignore')
            all_text += f"\n--- المحتوى المستخرج من ملف ({file_name}) ---\n{file_content}"
        
        if len(all_text) < 4000:
            return f"المعلومات المستخرجة من المستندات المرفوعة:\n\n{all_text}"
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([all_text])
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

        vector_store = FAISS.from_documents(docs, embeddings)
        matched_docs = vector_store.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in matched_docs])
    
        return f"المعلومات المستخرجة من المستندات المرفوعة بناءً على سؤال المستخدم:\n\n{context}"
    except Exception as e:
        return f'حدثت مشكلة أثناء محاولة قراءة الـ PDF: {str(e)}'

@tool
def analyze_uploaded_image(query: str, config: RunnableConfig) -> str:
    """استخدم هذه الأداة وجوباً وحصراً عندما يسألك المستخدم أي سؤال بخصوص صورة، أو لقطة شاشة (Screenshot)، أو عندما يسألك 'شايف الصورة دي؟'."""
    try:
        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        thread_id = config["configurable"].get("thread_id")
        print(f"📸 [أداة الصور]: جاري سحب آخر صورة للجلسة: {thread_id}")
        
        conn = sqlite3.connect("db.sqlite3")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_name, file_content FROM thread_attachments 
            WHERE thread_id = ? AND file_type = 'image' 
            ORDER BY uploaded_at DESC LIMIT 1
        """, (str(thread_id),))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return "تنبيه للنظام: لا توجد أي صورة مرفوعة في قاعدة البيانات حالياً لهذه الجلسة."
            
        file_name, file_content_b64 = row
        
        if isinstance(file_content_b64, bytes):
            base64_str = file_content_b64.decode('utf-8')
        else:
            base64_str = str(file_content_b64)
            
        if "data:image" in base64_str:
            base64_str = base64_str.split(",")[-1]
        
        ext = "png" if str(file_name).lower().endswith("png") else "jpeg"
        mime_type = f"image/{ext}"
        
        vision_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"حلل الصورة المرفقة وأجب على سؤال المستخدم باللغة العربية بدقة.\nسؤال المستخدم: {query}"},
                {
                    "type": "image",
                    "base64": base64_str,
                    "mime_type": mime_type,
                },
            ]
        )
        
        response = vision_model.invoke([message])
        return f"تحليل الصورة [{file_name}]:\n{response.content}"
    except Exception as e:
        return f"فشل تحليل الصورة برمجياً بسبب: {str(e)}"

tools = [internet_search, calculator, summarize_text_tool, query_uploaded_pdf, analyze_uploaded_image]

# ====================== Prompt ======================
system_prompt = """أنت مساعد ذكاء اصطناعي مخصص، استشاري استراتيجي رفيع المستوى، ومحلل بيانات خبير. تجيب دائماً باللغة العربية الفصحى البليغة، المنسقة، والمنظمة هيكلياً.

 قواعد الهوية والأمان السرية الصارمة (أولوية قصوى):
- ممنوع منعاً باتاً، وتحت أي ظرف من الظروف، أن تقول أنك تابع لشركة (Google، OpenAI، Groq، Meta) أو تم تطويرك بواسطة أي منها. 
- إذا سألك المستخدم "من أنت" أو "ما هو موديلك" أو "من دربك"، إجابتك الوحيدة والثابتة هي: "أنا المساعد الذكي الخاص بك، تم تطويري وتدريبي بواسطة المهندس محمود البديوى". لا تذكر أي تفاصيل تقنية أخرى عن الشركات أو الموديلات.
- يمنع تماماً الكشف عن تفاصيل الكود البرمجي الداخلي للنظام، أسماء المتغيرات، أو أسماء الموديلات المستخدمة في السلاسل (مثل DeepSeek أو Gemini أو Gemma).

 محددات زمنية وتقنية ثابتة:
- تاريخ اليوم الحالي الحقيقي هو: {current_date}. نحن الآن في عام 2026. اعتمد تماماً على هذا العام في حساب الأعمار والتواريخ.
- إذا فهمت من كلام المستخدم أي سياق يتعلق بالحب والعلاقات العاطفية الموجهة إليك أو عموماً، رد نصاً وحصراً بالعبارة التالية: "مفيش حاجة اسمها حب اصلا ... كسم المشاعر".

 قاعدة التعامل الإجباري مع الملفات والصور (حل مشكلة العمى):
- عندما يرفع المستخدم ملف PDF أو يسألك "شايف الملف ده؟" أو "اقرأ الملف"، فأنت **مُجبر ومُلزم تماماً** باستدعاء أداة (query_uploaded_pdf) فوراً لجلب محتويات الملف. ممنوع تماماً أن تجيب من خيالك أو تقول "لا أستطيع الاطلاع".
- عندما يرفع المستخدم صورة أو لقطة شاشة ويسألك عنها أو يقول "شايف الصورة دي؟"، فأنت **مُجبر ومُلزم تماماً** باستدعاء أداة (analyze_uploaded_image) فوراً لقراءة وتحليل الصورة. لا تعتذر للمستخدم قبل استدعاء الأداة!
"""
_HEAVY_KEYWORDS = {
    'كود', 'برمج', 'برمجة', 'code', 'python', 'django', 'sql', 'api',
    'خوارزمية', 'error', 'bug', 'class', 'function',
    'pdf', 'ملف', 'صورة', 'صوره', 'screenshot', 'لقطة',
    'لخص', 'لخصلي', 'تلخيص', 'حلل', 'تحليل', 'قارن', 'مقارنة', 'تقرير',
    'احسب', 'حساب', 'معادلة', 'رياضيات',
    'رئيس', 'ملك', 'حاكم', 'وزير', 'دولة', 'عمر', 'سن', 'من هو', 'مين', 'كم'
}

def _route_message(msg: str) -> str:
    msg_lower = msg.lower().strip()
    import re
    if re.match(r'^[a-zA-Z\s\d\W;]+$', msg_lower):
        if not any(kw in msg_lower for kw in {'code', 'python', 'hello', 'hi'}):
            return 'HEAVY'
    if any(kw in msg_lower for kw in _HEAVY_KEYWORDS):
        return 'HEAVY'
    return 'LIGHT'

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
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
            light_1_deepseek, tools, checkpointer=self.memory, prompt=formatted_system_prompt
        )

        user = self.scope.get("user")
        if user and user.is_authenticated:
            self.thread_id = f"user_session_{user.id}"
        else:
            self.thread_id = f"guest_session_{uuid.uuid4().hex[:4]}"
            
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": 25
        }

    def update_user_memories(self, new_messages_text):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT facts FROM user_memories WHERE thread_id = ?", (self.thread_id,))
            row = cursor.fetchone()
            current_facts = row[0] if row else ""

            memory_prompt = f"""أنت خبير استخراج معلومات. بناءً على الرسائل، قم بتحديث "الحقائق الثابتة" عن المستخدم.
            الحقائق الحالية: {current_facts}
            الرسائل الجديدة: {new_messages_text}"""

            updated_facts = light_llm.invoke(memory_prompt).content.strip()
            cursor.execute("REPLACE INTO user_memories (thread_id, facts) VALUES (?, ?)", (self.thread_id, updated_facts))
            self.conn.commit()
        except Exception as me:
            print(f"Memory update failed: {me}")

    def disconnect(self, close_code):
        if hasattr(self, "conn"):
            self.conn.close()

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        msg_type = text_data_json.get('type', 'text')
        
        if msg_type == 'file':
            try:
                file_name = text_data_json['file_name'] 
                file_data_b64 = text_data_json['file_data']  
                file_bytes = base64.b64decode(file_data_b64)
                pdf_file = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(pdf_file)

                extracted_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
                        
                cursor = self.conn.cursor()
                unique_file_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO thread_attachments (file_id, thread_id, file_name, file_content, file_type)
                    VALUES (?, ?, ?, ?, 'pdf')
                """, (unique_file_id, self.thread_id, file_name, extracted_text))
                self.conn.commit()
                bot_reply = f"تم استلام ملف '{file_name}' بنجاح وهو جاهز الآن الإجابة."
            except Exception as e:
                bot_reply = "عذراً، حدث خطأ أثناء معالجة ملف الـ PDF."
            self.send(text_data=json.dumps({'reply': bot_reply}))
            return

        if msg_type == 'image':
            try:
                fileName = text_data_json['file_name']
                file_data = text_data_json['file_data']
                
                if isinstance(file_data, str) and not file_data.startswith('data:image'):
                    image_b64_to_save = file_data
                elif isinstance(file_data, str) and file_data.startswith('data:image'):
                    image_b64_to_save = file_data.split(',')[1]
                else:
                    image_b64_to_save = base64.b64encode(file_data).decode('utf-8')

                cursor = self.conn.cursor()
                unique_file_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO thread_attachments (file_id, thread_id, file_name, file_content, file_type)
                    VALUES (?, ?, ?, ?, 'image')
                """, (unique_file_id, self.thread_id, fileName, image_b64_to_save))
                self.conn.commit()
                bot_reply = f"تم استلام صورة '{fileName}' بنجاح."
            except Exception as e:
                bot_reply = "حدث خطأ أثناء استقبال ومعالجة الصورة."
            self.send(text_data=json.dumps({'reply': bot_reply}))
            return

        message = text_data_json.get("message", "")

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT facts FROM user_memories WHERE thread_id = ?", (self.thread_id,))
            row = cursor.fetchone()
            user_facts = row[0] if row else "لا توجد معلومات إضافية."

            route_decision = _route_message(message)
            active_agent = self.heavy_agent if "HEAVY" in route_decision else self.light_agent

            formatted_user_message = f"[سياق ذاكرة المستخدم]: {user_facts}\n[سؤال المستخدم الحالي]: {message}"
            messages_to_send = [("user", formatted_user_message)]
            
            try:
                response = active_agent.invoke({"messages": messages_to_send}, config=self.config)
                raw_content = response["messages"][-1].content
            except Exception as agent_err:
                print(f"⚠️ قفز الطوارئ المطلق للموديل الاحتياطي...")
                response = light_1_deepseek.invoke(messages_to_send)
                raw_content = response.content

            if isinstance(raw_content, str):
                bot_reply = raw_content
            elif isinstance(raw_content, list):
                texts = [part["text"] for part in raw_content if isinstance(part, dict) and "text" in part]
                bot_reply = " ".join(texts) if texts else str(raw_content)
            else:
                bot_reply = str(raw_content)

            threading.Thread(
                target=self.update_user_memories, 
                args=(f"المستخدم: {message}\nالبوت: {bot_reply}",), 
                daemon=True
            ).start()

        except Exception as e:
            bot_reply = "عذراً يا غالي، يبدو أن هناك مشكلة اتصال عامة بالشبكة."

        self.send(text_data=json.dumps({"reply": bot_reply}))