from django.urls import path
from ai import views

urlpatterns = [
    path('login/', views.login_page, name='login'),
    path('api/login/', views.login_api, name='login_api'),
    path('', views.chat_page, name='chat'), 

    path('register/', views.register_page, name='register'),
    path('api/register/', views.register_api, name='register_api'),
    path('api/delete-chat/', views.delete_chat_api, name='delete_chat_api'),]
handler404 = 'django.views.defaults.page_not_found'