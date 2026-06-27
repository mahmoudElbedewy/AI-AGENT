from django.urls import path
from ai import views
from .views import (
    share_conversation_api,
    get_shared_conversation_api,
    shared_conversation_page,
)

urlpatterns = [
    path("login/", views.login_page, name="login"),
    path("api/login/", views.login_api, name="login_api"),
    path("", views.chat_page, name="chat"),
    path("register/", views.register_page, name="register"),
    path("api/register/", views.register_api, name="register_api"),
    path("api/delete-chat/", views.delete_chat_api, name="delete_chat_api"),
    path("api/share/", share_conversation_api, name="share_conversation"),
    path(
        "api/shared/<str:share_token>/",
        get_shared_conversation_api,
        name="get_shared_conversation",
    ),
    path(
        "shared/<str:share_token>/", shared_conversation_page, name="shared_chat_page"
    ),
]
handler404 = "django.views.defaults.page_not_found"
