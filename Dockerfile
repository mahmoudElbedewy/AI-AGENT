FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# عمل الـ migrations لتهيئة جداول قاعدة البيانات قبل التشغيل
RUN python manage.py migrate

EXPOSE 7860

# تشغيل daphne ليدعم الـ WebSockets (wss) على Hugging Face
CMD ["daphne", "-b", "0.0.0.0", "-p", "7860", "core.asgi:application"]