web: sh -c "python manage.py migrate --noinput && gunicorn modules.dataanalyzer.wsgi:application --bind 0.0.0.0:$PORT"
