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