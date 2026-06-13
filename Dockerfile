FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV DJANGO_SUPERUSER_USERNAME=m
ENV DJANGO_SUPERUSER_EMAIL=m@m.com
ENV DJANGO_SUPERUSER_PASSWORD=m

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 7860

CMD python manage.py migrate && \
    python manage.py createsuperuser --noinput || true && \
    python -m daphne -b 0.0.0.0 -p 7860 core.asgi:application