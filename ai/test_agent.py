# -*- coding: utf-8 -*-
"""
تيستات شاملة للأجنت (consumers.py).

تغطي:
  1. منطق المحادثات المتعددة + الذاكرة المشتركة (DB layer) - TestMultiConversation
  2. دوال التطبيع وكشف اللغة العربية - TestNormalization
  3. دوال تصنيف نوايا الرسائل (routing logic) - TestIntentDetection
     - PDF/سيرة ذاتية
     - تنفيذ كود
     - إنشاء ملفات
     - توليد صور (+ متابعات)
     - بحث لايف (live search)
     - طلبات تلخيص
     - طلب جدول
     - رفع ملف بدون رسالة فعلية
  4. كشف تسريب التفكير الداخلي للبوت - TestInternalLeakDetection
  5. استخراج الكلمات المفتاحية للبحث في PDF - TestQueryTerms

طريقة التشغيل:
    cd /home/claude/test_project
    python3 -m unittest test_agent_full -v

ملحوظة: التيستات دي مبنية على استخراج الدوال "النقية" (pure functions) من
consumers.py عبر load_helpers.py، بدون الحاجة لتشغيل Django/Channels/LLM APIs،
وعلى محاكاة طبقة قاعدة البيانات (SQLite) لاختبار منطق المحادثات والذاكرة.
"""

import os
import sqlite3
import tempfile
import unittest
import uuid

from load_helpers import load_helpers

H = load_helpers()

_normalize_arabic_query = H["_normalize_arabic_query"]
_is_pdf_related_question = H["_is_pdf_related_question"]
_needs_code_execution = H["_needs_code_execution"]
_needs_file_creation = H["_needs_file_creation"]
_needs_image_generation = H["_needs_image_generation"]
_is_image_generation_followup = H["_is_image_generation_followup"]
_needs_live_search = H["_needs_live_search"]
_is_summary_request = H["_is_summary_request"]
_wants_table = H["_wants_table"]
_is_arabic_text = H["_is_arabic_text"]
_has_actionable_upload_request = H["_has_actionable_upload_request"]
_contains_internal_leak = H["_contains_internal_leak"]
_query_terms = H["_query_terms"]


# ======================================================================
# 1) منطق المحادثات المتعددة + الذاكرة المشتركة (نفس DB layer المعدّل)
# ======================================================================

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS thread_attachments (
            file_id TEXT PRIMARY KEY,
            thread_id TEXT,
            file_name TEXT,
            file_content TEXT,
            file_type TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            user_id TEXT PRIMARY KEY,
            facts TEXT DEFAULT ''
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT DEFAULT 'محادثة جديدة',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            role TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


class FakeConsumer:
    """نسخة مصغّرة من المنطق المعدّل في ChatConsumer للاختبار بدون Django/Channels."""

    def __init__(self, db_path, user_id, conversation_id=None):
        self.db = db_path
        self.user_id = user_id
        is_new = False
        if not conversation_id:
            conversation_id = uuid.uuid4().hex
            is_new = True
        self.conversation_id = conversation_id
        self.thread_id = f"{self.user_id}_conv_{conversation_id}"
        if is_new:
            self._create_conversation_record()

    def _create_conversation_record(self):
        with sqlite3.connect(self.db, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO conversations (conversation_id, user_id, title)
                VALUES (?, ?, ?)
                """,
                (self.conversation_id, self.user_id, "محادثة جديدة"),
            )
            conn.commit()

    def load_conversations_list(self):
        with sqlite3.connect(self.db, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT conversation_id, title, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (self.user_id,),
            )
            return [
                {"conversation_id": r[0], "title": r[1], "updated_at": r[2]}
                for r in cur.fetchall()
            ]

    def save_chat_message(self, role, message):
        if not message or not str(message).strip():
            return
        with sqlite3.connect(self.db, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_messages (id, thread_id, role, message)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), self.thread_id, role, str(message)),
            )
            if role == "user":
                cur.execute(
                    "SELECT title FROM conversations WHERE conversation_id = ?",
                    (self.conversation_id,),
                )
                row = cur.fetchone()
                if row and (row[0] == "محادثة جديدة" or not row[0]):
                    new_title = str(message).strip()[:50]
                    cur.execute(
                        "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                        (new_title, self.conversation_id),
                    )
                else:
                    cur.execute(
                        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                        (self.conversation_id,),
                    )
            conn.commit()

    def load_chat_history(self):
        with sqlite3.connect(self.db, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT role, message FROM chat_messages
                WHERE thread_id = ?
                ORDER BY created_at ASC
                """,
                (self.thread_id,),
            )
            history = []
            for role, message in cur.fetchall():
                if role == "bot" and _contains_internal_leak(message):
                    continue
                history.append({"role": role, "message": message})
            return history

    def update_user_memories(self, new_messages_text):
        with sqlite3.connect(self.db, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT facts FROM user_memories WHERE user_id = ?",
                (self.user_id,),
            )
            row = cur.fetchone()
            current_facts = row[0].strip() if row and row[0] else ""
            new_fact = str(new_messages_text).strip()[:4000]
            updated_facts = (
                f"{current_facts}\n\n{new_fact}".strip()
                if current_facts
                else new_fact
            )
            cur.execute(
                "REPLACE INTO user_memories (user_id, facts) VALUES (?, ?)",
                (self.user_id, updated_facts),
            )
            conn.commit()

    def get_user_facts(self):
        with sqlite3.connect(self.db, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT facts FROM user_memories WHERE user_id = ?",
                (self.user_id,),
            )
            row = cur.fetchone()
            return row[0] if row else "No historic context data found yet."


class TestMultiConversation(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        init_db(self.db_path)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except PermissionError:
            pass

    def test_new_conversation_creates_unique_thread_id(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c2 = FakeConsumer(self.db_path, user_id="user_1")
        self.assertNotEqual(c1.thread_id, c2.thread_id)
        self.assertTrue(c1.thread_id.startswith("user_1_conv_"))
        self.assertTrue(c2.thread_id.startswith("user_1_conv_"))

    def test_resuming_conversation_uses_same_thread_id(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c1.save_chat_message("user", "أول رسالة")
        c1_resumed = FakeConsumer(self.db_path, user_id="user_1", conversation_id=c1.conversation_id)
        self.assertEqual(c1.thread_id, c1_resumed.thread_id)
        history = c1_resumed.load_chat_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["message"], "أول رسالة")

    def test_conversations_list_per_user(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c2 = FakeConsumer(self.db_path, user_id="user_1")
        other_user = FakeConsumer(self.db_path, user_id="user_2")

        convos = c1.load_conversations_list()
        ids = [c["conversation_id"] for c in convos]
        self.assertIn(c1.conversation_id, ids)
        self.assertIn(c2.conversation_id, ids)
        self.assertEqual(len(convos), 2)

        other_convos = other_user.load_conversations_list()
        self.assertEqual(len(other_convos), 1)
        self.assertNotIn(c1.conversation_id, [c["conversation_id"] for c in other_convos])

    def test_chat_history_isolated_between_conversations(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c2 = FakeConsumer(self.db_path, user_id="user_1")

        c1.save_chat_message("user", "رسالة في الشات الأول")
        c2.save_chat_message("user", "رسالة في الشات الثاني")

        self.assertEqual(len(c1.load_chat_history()), 1)
        self.assertEqual(len(c2.load_chat_history()), 1)
        self.assertEqual(c1.load_chat_history()[0]["message"], "رسالة في الشات الأول")
        self.assertEqual(c2.load_chat_history()[0]["message"], "رسالة في الشات الثاني")

    def test_user_memories_shared_across_conversations(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c2 = FakeConsumer(self.db_path, user_id="user_1")

        c1.update_user_memories("User: اسمي أحمد\nBot: تشرفت أحمد")

        facts_from_c2 = c2.get_user_facts()
        self.assertIn("أحمد", facts_from_c2)

    def test_user_memories_isolated_between_users(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c2 = FakeConsumer(self.db_path, user_id="user_2")

        c1.update_user_memories("User: اسمي أحمد")
        c2.update_user_memories("User: اسمي سارة")

        self.assertIn("أحمد", c1.get_user_facts())
        self.assertNotIn("سارة", c1.get_user_facts())
        self.assertIn("سارة", c2.get_user_facts())
        self.assertNotIn("أحمد", c2.get_user_facts())

    def test_memories_append_not_overwrite(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c1.update_user_memories("معلومة 1")
        c1.update_user_memories("معلومة 2")
        facts = c1.get_user_facts()
        self.assertIn("معلومة 1", facts)
        self.assertIn("معلومة 2", facts)

    def test_conversation_title_set_from_first_user_message(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c1.save_chat_message("user", "ازاي أتعلم بايثون من الصفر؟")

        convos = c1.load_conversations_list()
        this_convo = next(c for c in convos if c["conversation_id"] == c1.conversation_id)
        self.assertEqual(this_convo["title"], "ازاي أتعلم بايثون من الصفر؟")

    def test_conversation_title_not_overwritten_by_later_messages(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c1.save_chat_message("user", "أول سؤال")
        c1.save_chat_message("bot", "رد البوت")
        c1.save_chat_message("user", "سؤال تاني")

        convos = c1.load_conversations_list()
        this_convo = next(c for c in convos if c["conversation_id"] == c1.conversation_id)
        self.assertEqual(this_convo["title"], "أول سؤال")

    def test_new_conversation_default_title(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        convos = c1.load_conversations_list()
        this_convo = next(c for c in convos if c["conversation_id"] == c1.conversation_id)
        self.assertEqual(this_convo["title"], "محادثة جديدة")

    def test_guest_users_isolated(self):
        g1 = FakeConsumer(self.db_path, user_id="guest_aaaa")
        g2 = FakeConsumer(self.db_path, user_id="guest_bbbb")
        g1.save_chat_message("user", "رسالة ضيف 1")
        g2.save_chat_message("user", "رسالة ضيف 2")

        self.assertEqual(len(g1.load_chat_history()), 1)
        self.assertEqual(len(g2.load_chat_history()), 1)
        self.assertNotEqual(g1.thread_id, g2.thread_id)

    def test_reconnect_without_conversation_id_creates_new(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c2 = FakeConsumer(self.db_path, user_id="user_1")
        self.assertNotEqual(c1.conversation_id, c2.conversation_id)
        self.assertEqual(len(c1.load_conversations_list()), 2)

    def test_bot_messages_with_internal_leak_excluded_from_history(self):
        c1 = FakeConsumer(self.db_path, user_id="user_1")
        c1.save_chat_message("user", "سؤال عادي")
        c1.save_chat_message("bot", "Based on the user's request, I will analyze this.")
        c1.save_chat_message("bot", "إجابة طبيعية")

        history = c1.load_chat_history()
        bot_messages = [h["message"] for h in history if h["role"] == "bot"]
        self.assertEqual(bot_messages, ["إجابة طبيعية"])


# ======================================================================
# 2) دوال التطبيع وكشف اللغة العربية
# ======================================================================

class TestNormalization(unittest.TestCase):
    def test_normalize_alef_variants(self):
        self.assertEqual(_normalize_arabic_query("أحمد"), "احمد")
        self.assertEqual(_normalize_arabic_query("إحمد"), "احمد")
        self.assertEqual(_normalize_arabic_query("آحمد"), "احمد")

    def test_normalize_alef_maksura_and_taa_marbouta(self):
        self.assertEqual(_normalize_arabic_query("على"), "علي")
        self.assertEqual(_normalize_arabic_query("مدرسة"), "مدرسه")

    def test_normalize_removes_diacritics(self):
        # تشكيل (فتحة وضمة وكسرة)
        self.assertEqual(_normalize_arabic_query("مُحَمَّدٌ"), "محمد")

    def test_normalize_lowercases_english(self):
        self.assertEqual(_normalize_arabic_query("HELLO World"), "hello world")

    def test_normalize_strips_whitespace(self):
        self.assertEqual(_normalize_arabic_query("   مرحبا   "), "مرحبا")

    def test_normalize_empty_input(self):
        self.assertEqual(_normalize_arabic_query(""), "")
        self.assertEqual(_normalize_arabic_query(None), "")

    def test_is_arabic_text_true(self):
        self.assertTrue(_is_arabic_text("مرحبا"))
        self.assertTrue(_is_arabic_text("hello مرحبا"))

    def test_is_arabic_text_false(self):
        self.assertFalse(_is_arabic_text("hello world"))
        self.assertFalse(_is_arabic_text(""))
        self.assertFalse(_is_arabic_text(None))


# ======================================================================
# 3) دوال تصنيف نوايا الرسائل (Routing Logic)
# ======================================================================

class TestPdfRelatedDetection(unittest.TestCase):
    def test_cv_arabic_variants(self):
        self.assertTrue(_is_pdf_related_question("ابعت السيفي بتاعي"))
        self.assertTrue(_is_pdf_related_question("اعرفني عن نفسي من الملف"))
        self.assertTrue(_is_pdf_related_question("انا مين على حسب الملف ده"))

    def test_cv_english(self):
        self.assertTrue(_is_pdf_related_question("can you read my resume?"))
        self.assertTrue(_is_pdf_related_question("summarize this PDF"))
        self.assertTrue(_is_pdf_related_question("what's in my profile"))

    def test_unrelated_message(self):
        self.assertFalse(_is_pdf_related_question("ايه أخبارك عامل ايه"))
        self.assertFalse(_is_pdf_related_question("hello how are you"))

    def test_book_lecture_keywords(self):
        self.assertTrue(_is_pdf_related_question("لخصلي الكتاب ده"))

    def test_lecture_keyword_normalization_mismatch(self):
        # ملحوظة: كلمة "محاضرة" في قائمة الكلمات المفتاحية مكتوبة بـ "ة"،
        # لكن دالة التطبيع بتحول "ة" إلى "ه" دايماً، فالكلمة المطبّعة
        # "المحاضره" لا تطابق "محاضرة" أبداً. ده bug في الكود الحالي
        # موجود هنا لتوثيق السلوك الفعلي (متوقع False دلوقتي).
        self.assertFalse(_is_pdf_related_question("اشرحلي المحاضرة"))


class TestCodeExecutionDetection(unittest.TestCase):
    def test_arabic_run_code(self):
        self.assertTrue(_needs_code_execution("شغل الكود ده"))
        self.assertTrue(_needs_code_execution("جرب السكريبت دا"))

    def test_english_run_code(self):
        self.assertTrue(_needs_code_execution("run this python script"))
        self.assertTrue(_needs_code_execution("execute this function"))

    def test_trigger_without_code_word(self):
        # فيه تريجر "شغل" بس مفيش كلمة كود
        self.assertFalse(_needs_code_execution("شغل التلفزيون"))

    def test_code_word_without_trigger(self):
        self.assertFalse(_needs_code_execution("الكود ده حلو"))

    def test_unrelated(self):
        self.assertFalse(_needs_code_execution("ازيك عامل ايه"))


class TestFileCreationDetection(unittest.TestCase):
    def test_word_doc_arabic(self):
        self.assertTrue(_needs_file_creation("اعملي تقرير وورد"))
        self.assertTrue(_needs_file_creation("جهزلي سي في"))

    def test_excel_arabic(self):
        self.assertTrue(_needs_file_creation("انشئ ملف اكسل بالمصاريف"))

    def test_english(self):
        self.assertTrue(_needs_file_creation("create a resume for me"))
        self.assertTrue(_needs_file_creation("generate an excel report"))

    def test_trigger_without_filetype(self):
        self.assertFalse(_needs_file_creation("اعملي أكل"))

    def test_filetype_without_trigger(self):
        self.assertFalse(_needs_file_creation("الورد بتاعي ضاع"))


class TestImageGenerationDetection(unittest.TestCase):
    def test_arabic_draw_requests(self):
        self.assertTrue(_needs_image_generation("ارسملي قطة"))
        self.assertTrue(_needs_image_generation("اعمل صورة لمنظر طبيعي"))
        self.assertTrue(_needs_image_generation("ولد صورة لسيارة"))

    def test_english_draw_requests(self):
        self.assertTrue(_needs_image_generation("draw a sunset"))
        self.assertTrue(_needs_image_generation("generate image of a cat"))
        self.assertTrue(_needs_image_generation("create a picture of mountains"))

    def test_unrelated(self):
        self.assertFalse(_needs_image_generation("ازاي اعمل اكل المكرونة"))
        self.assertFalse(_needs_image_generation("hello there"))


class TestImageGenerationFollowup(unittest.TestCase):
    def test_followup_confirmation_after_bot_offer(self):
        last_bot = "تحب أوصفلك الصورة وأعملها؟"
        self.assertTrue(_is_image_generation_followup("تمام", last_bot))
        self.assertTrue(_is_image_generation_followup("ايوه ابعتلك", last_bot))

    def test_followup_with_image_word(self):
        last_bot = "هل تريد أن أعمل لك صورة؟"
        self.assertTrue(_is_image_generation_followup("صورة لكلب جالس", last_bot))

    def test_no_followup_without_bot_offer(self):
        last_bot = "رد عادي من البوت مالوش علاقة بالصور"
        self.assertFalse(_is_image_generation_followup("تمام", last_bot))

    def test_no_followup_without_confirmation_or_image_ref(self):
        last_bot = "هل تريد أن أعمل لك صورة؟"
        self.assertFalse(_is_image_generation_followup("ما رأيك في الطقس؟", last_bot))


class TestLiveSearchDetection(unittest.TestCase):
    def test_finance_keywords(self):
        self.assertTrue(_needs_live_search("سعر الدولار النهاردة"))
        self.assertTrue(_needs_live_search("سعر الذهب دلوقتي"))

    def test_news_politics_keywords(self):
        self.assertTrue(_needs_live_search("مين رئيس مصر الحالي"))
        self.assertTrue(_needs_live_search("اخر اخبار اليوم"))

    def test_sports_keywords(self):
        self.assertTrue(_needs_live_search("نتيجة مباراة الاهلي امبارح"))

    def test_weather_keywords(self):
        self.assertTrue(_needs_live_search("الطقس بكرة عامل ايه"))

    def test_english_live_patterns(self):
        self.assertTrue(_needs_live_search("what is the latest news today"))
        self.assertTrue(_needs_live_search("current bitcoin price"))

    def test_year_pattern(self):
        from datetime import datetime
        year = datetime.now().year
        self.assertTrue(_needs_live_search(f"events happening in {year}"))

    def test_no_live_search_needed(self):
        self.assertFalse(_needs_live_search("ازاي اعمل عجة بالبيض"))
        self.assertFalse(_needs_live_search("explain photosynthesis to me"))

    def test_what_is_pattern_triggers_live_search(self):
        # ملحوظة: "what is" موجودة في english_live_patterns، فأي سؤال
        # يبدأ بـ "what is" بيتحول لبحث لايف حتى لو الموضوع ثابت علمياً
        # (مثال: "what is photosynthesis"). ده سلوك حالي للكود قد يحتاج
        # مراجعة لاحقاً، بس التيست ده موجود لتوثيق السلوك الفعلي.
        self.assertTrue(_needs_live_search("what is photosynthesis"))

    def test_empty_message(self):
        self.assertFalse(_needs_live_search(""))
        self.assertFalse(_needs_live_search(None))


class TestSummaryRequestDetection(unittest.TestCase):
    def test_arabic_summary_terms(self):
        self.assertTrue(_is_summary_request("لخصلي الموضوع ده"))
        self.assertTrue(_is_summary_request("اعمل ملخص للمحاضرة"))
        self.assertTrue(_is_summary_request("اهم النقاط في الكتاب"))

    def test_english_summary_terms(self):
        self.assertTrue(_is_summary_request("can you summarize this article"))
        self.assertTrue(_is_summary_request("give me the key points"))

    def test_not_a_summary_request(self):
        self.assertFalse(_is_summary_request("ايه رأيك في الموضوع"))
        self.assertFalse(_is_summary_request("hello"))


class TestWantsTableDetection(unittest.TestCase):
    def test_arabic_table_request(self):
        self.assertTrue(_wants_table("اعرض البيانات في جدول"))
        self.assertTrue(_wants_table("حولها على شكل جدول"))

    def test_english_table_request(self):
        self.assertTrue(_wants_table("show this in a table"))
        self.assertTrue(_wants_table("give me tabular data"))

    def test_no_table_request(self):
        self.assertFalse(_wants_table("اعمل ليا تقرير عادي"))
        self.assertFalse(_wants_table("hello world"))


class TestActionableUploadRequest(unittest.TestCase):
    def test_pure_filler_messages_are_not_actionable(self):
        self.assertFalse(_has_actionable_upload_request("اتفضل"))
        self.assertFalse(_has_actionable_upload_request("ده الملف"))
        self.assertFalse(_has_actionable_upload_request("photo"))

    def test_empty_message_not_actionable(self):
        self.assertFalse(_has_actionable_upload_request(""))
        self.assertFalse(_has_actionable_upload_request("   "))

    def test_real_question_is_actionable(self):
        self.assertTrue(_has_actionable_upload_request("لخصلي محتوى الملف ده"))
        self.assertTrue(_has_actionable_upload_request("what does this document say about pricing?"))


# ======================================================================
# 4) كشف تسريب التفكير الداخلي للبوت
# ======================================================================

class TestInternalLeakDetection(unittest.TestCase):
    def test_detects_leak_patterns(self):
        self.assertTrue(_contains_internal_leak("Based on the user's request, I will do X"))
        self.assertTrue(_contains_internal_leak("[User Query Analysis] the user wants..."))
        self.assertTrue(_contains_internal_leak("I'll try to analyze the question first"))
        self.assertTrue(_contains_internal_leak("To clarify, the user means something else"))

    def test_normal_reply_not_flagged(self):
        self.assertFalse(_contains_internal_leak("أهلاً بيك! إزاي أقدر أساعدك النهاردة؟"))
        self.assertFalse(_contains_internal_leak("الإجابة هي 42."))

    def test_empty_text(self):
        self.assertFalse(_contains_internal_leak(""))
        self.assertFalse(_contains_internal_leak(None))


# ======================================================================
# 5) استخراج الكلمات المفتاحية (لاستخدامها في البحث داخل ملفات PDF)
# ======================================================================

class TestQueryTerms(unittest.TestCase):
    def test_filters_stopwords(self):
        terms = _query_terms("ما هو اسمي في الملف ده؟")
        self.assertNotIn("ما", terms)
        self.assertNotIn("هو", terms)
        self.assertNotIn("في", terms)
        self.assertIn("اسمي", terms)

    def test_filters_short_tokens(self):
        terms = _query_terms("a an of in to me my")
        self.assertEqual(terms, [])

    def test_extracts_meaningful_terms(self):
        terms = _query_terms("ما هي خبرتي في مجال البرمجة")
        self.assertIn("خبرتي", terms)
        self.assertIn("مجال", terms)
        self.assertIn("البرمجه", terms)  # ة -> ه بعد التطبيع


if __name__ == "__main__":
    unittest.main(verbosity=2)