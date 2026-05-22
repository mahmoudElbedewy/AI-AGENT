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


EXPOSE 7860
CMD python manage.py migrate && \
    python manage.py shell -c "from django.contrib.auth.models import User; User.objects.filter(username='m').exists() or User.objects.create_superuser('m', 'm@m.com', 'm')" && \
    daphne -b 0.0.0.0 -p 7860 core.asgi:application