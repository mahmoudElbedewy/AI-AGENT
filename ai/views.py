from django.shortcuts import render

def chat_with_agent(request):
    # الدالة دي وظيفتها الوحيدة إنها تعرض واجهة الشات
    # الذكاء الاصطناعي والردود كلها شغالة في consumers.py
    return render(request, 'chat_template.html')