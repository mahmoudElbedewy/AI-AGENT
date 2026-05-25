"""
ASGI config for core project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

import os
from channels.routing import URLRouter , ProtocolTypeRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

django.setup()
django_asgi_app = get_asgi_application()

import ai.routing
from .middleware import JWTAuthMiddleware
application = ProtocolTypeRouter({
    'http' : get_asgi_application(),

    'websocket' : JWTAuthMiddleware(
        URLRouter(ai.routing.websocket_urlpatterns)
    )
})
