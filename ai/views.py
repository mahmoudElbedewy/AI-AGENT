import json
import jwt
from datetime import datetime, timedelta
from django.conf import settings
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .forms import StrictRegistrationForm
import sqlite3
import secrets
from django.core.cache import cache



def login_page(request):
    return render(request, "login.html")

def chat_page(request):
    return render(request, "chat_template.html")

def register_page(request):
    return render(request, "register.html")

@csrf_exempt
def login_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            username = data.get("username")
            password = data.get("password")
            
            user = authenticate(username=username, password=password)
            if user is not None:
                payload = {
                    "user_id": user.id,
                    "exp": datetime.utcnow() + timedelta(days=7)
                }
                token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
                
                return JsonResponse({
                    "access_token": token,
                    "user_data": {
                        "name": user.first_name if user.first_name else user.username
                    }
                })
            else:
                return JsonResponse({"error": "اسم المستخدم أو كلمة المرور غير صحيحة"}, status=400)
        except Exception as e:
            return JsonResponse({"error": "حدث خطأ في الخادم"}, status=500)
            
    return JsonResponse({"error": "Method not allowed"}, status=405)
    

@csrf_exempt
def register_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            form = StrictRegistrationForm(data)
            
            if form.is_valid():
                user = form.save(commit=False)
                user.set_password(form.cleaned_data['password'])
                user.save()
                
                payload = {
                    "user_id": user.id,
                    "exp": datetime.utcnow() + timedelta(days=7)
                }
                token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
                
                return JsonResponse({
                    "access_token": token,
                    "user_data": {
                        "name": user.username
                    }
                })
            else:
                error_msg = "خطأ في البيانات المُدخلة"
                for field, errors in form.errors.items():
                    error_msg = errors[0]
                    break
                return JsonResponse({"error": error_msg}, status=400)
                
        except Exception as e:
            return JsonResponse({"error": f"حدث خطأ في الخادم: {str(e)}"}, status=500)
            
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def delete_chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            token = data.get("thread_id") 
            
            if not token:
                return JsonResponse({"error": "التوكن غير موجود"}, status=400)

            try:
                payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
                user_id = payload.get("user_id")
            except Exception as e:
                return JsonResponse({"error": "التوكن غير صالح أو منتهي الصلاحية"}, status=400)

            if not user_id:
                return JsonResponse({"error": "بيانات المستخدم غير صالحة"}, status=400)

            actual_thread_id = f"user_session_{user_id}"

            with sqlite3.connect("db.sqlite3", timeout=20) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_messages WHERE thread_id = ?", (actual_thread_id,))
                cursor.execute("DELETE FROM user_memories WHERE thread_id = ?", (actual_thread_id,))
                cursor.execute("DELETE FROM thread_attachments WHERE thread_id = ?", (actual_thread_id,))
                conn.commit()

            return JsonResponse({"success": "تم حذف المحادثة بالكامل بنجاح"})
        except Exception as e:
            return JsonResponse({"error": f"حدث خطأ أثناء الحذف: {str(e)}"}, status=500)
            
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def share_conversation_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        token = data.get("token")
        conversation_id = data.get("conversation_id")

        if not token or not conversation_id:
            return JsonResponse({"error": "بيانات ناقصة"}, status=400)

        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("user_id")
        except Exception:
            return JsonResponse({"error": "توكن غير صالح"}, status=401)

        if not user_id:
            return JsonResponse({"error": "مستخدم غير صالح"}, status=401)

        db_user_id = f"user_{user_id}"

        with sqlite3.connect("db.sqlite3", timeout=20) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT title FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, db_user_id),
            )
            conv_row = cursor.fetchone()
            if not conv_row:
                return JsonResponse({"error": "المحادثة مش موجودة"}, status=404)

            conv_title = conv_row[0] or "محادثة مشتركة"
            thread_id = f"{db_user_id}_conv_{conversation_id}"

            cursor.execute(
                """SELECT role, message FROM chat_messages
                   WHERE thread_id = ?
                   ORDER BY created_at ASC""",
                (thread_id,),
            )
            messages = [
                {"role": row[0], "message": row[1]}
                for row in cursor.fetchall()
                if row[0] in ("user", "bot")
            ]

            if not messages:
                return JsonResponse({"error": "المحادثة فاضية"}, status=400)

            share_token = secrets.token_urlsafe(20)
            expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_conversations (
                    share_token TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_id TEXT,
                    title TEXT,
                    messages_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT
                )
            """)
            cursor.execute(
                """INSERT OR REPLACE INTO shared_conversations
                   (share_token, conversation_id, user_id, title, messages_json, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    share_token,
                    conversation_id,
                    db_user_id,
                    conv_title,
                    json.dumps(messages, ensure_ascii=False),
                    expires_at,
                ),
            )
            conn.commit()

        share_url = f"/shared/{share_token}/"
        return JsonResponse({
            "share_token": share_token,
            "share_url": share_url,
            "title": conv_title,
            "expires_at": expires_at,
        })

    except Exception as e:
        return JsonResponse({"error": f"خطأ في الخادم: {str(e)}"}, status=500)


@csrf_exempt
def get_shared_conversation_api(request, share_token):
    """
    بيرجع محتوى المحادثة المشتركة (بدون ما يحتاج login).
    """
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        with sqlite3.connect("db.sqlite3", timeout=20) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_conversations (
                    share_token TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_id TEXT,
                    title TEXT,
                    messages_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT
                )
            """)
            cursor.execute(
                "SELECT title, messages_json, expires_at FROM shared_conversations WHERE share_token = ?",
                (share_token,),
            )
            row = cursor.fetchone()

        if not row:
            return JsonResponse({"error": "الرابط ده مش موجود أو انتهت صلاحيته"}, status=404)

        title, messages_json, expires_at = row

        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at)
                if datetime.utcnow() > exp:
                    return JsonResponse({"error": "انتهت صلاحية هذا الرابط"}, status=410)
            except Exception:
                pass

        messages = json.loads(messages_json) if messages_json else []
        return JsonResponse({
            "title": title,
            "messages": messages,
        })

    except Exception as e:
        return JsonResponse({"error": f"خطأ: {str(e)}"}, status=500)


def shared_conversation_page(request, share_token):
    """
    بيعرض صفحة HTML بسيطة للمحادثة المشتركة.
    """
    from django.shortcuts import render
    return render(request, "shared_chat.html", {"share_token": share_token})


def export_messages(request):
    with sqlite3.connect("db.sqlite3") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, role, message, created_at FROM chat_messages ORDER BY created_at ASC"
        )
        rows = cursor.fetchall()
    
    data = [
        {"thread_id": r[0], "role": r[1], "message": r[2], "created_at": r[3]}
        for r in rows
    ]
    return JsonResponse({"messages": data}, json_dumps_params={"ensure_ascii": False})