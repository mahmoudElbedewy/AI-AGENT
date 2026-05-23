from django.shortcuts import render

def chat_with_agent(request):
    return render(request, 'chat_template.html')