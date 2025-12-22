FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (matplotlib fonts/backends)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     gcc \
     g++ \
     libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV DJANGO_SETTINGS_MODULE=dataanalyzer.settings

# Collect static at build time (can be skipped if you prefer runtime)
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD gunicorn dataanalyzer.wsgi:application --bind 0.0.0.0:${PORT:-8000}
