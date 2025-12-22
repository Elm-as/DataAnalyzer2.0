from pathlib import Path
import os
from urllib.parse import urlparse

# settings.py est dans modules/dataanalyzer/, on remonte à la racine du repo
BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-only-secret-key-change-me')


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


DEBUG = _env_bool('DJANGO_DEBUG', default=False)

# ALLOWED_HOSTS
# - En local: autorise localhost
# - En prod (Railway): préfère une liste explicite via DJANGO_ALLOWED_HOSTS
# - Fallback: détecte le domaine public Railway si présent
allowed_hosts: list[str] = ['127.0.0.1', 'localhost']

_hosts = os.environ.get('DJANGO_ALLOWED_HOSTS')
if _hosts:
    allowed_hosts.extend([h.strip() for h in _hosts.split(',') if h.strip()])

_railway_public_domain = os.environ.get('RAILWAY_PUBLIC_DOMAIN')
if _railway_public_domain:
    allowed_hosts.append(_railway_public_domain.strip())
else:
    # Certains environnements exposent une URL; on en extrait le host
    for _url_var in ('RAILWAY_STATIC_URL', 'RAILWAY_URL', 'PUBLIC_URL', 'APP_URL'):
        _url_val = os.environ.get(_url_var)
        if not _url_val:
            continue
        try:
            _parsed = urlparse(_url_val)
            if _parsed.hostname:
                allowed_hosts.append(_parsed.hostname)
                break
        except Exception:
            continue

# Dé-duplique en conservant l'ordre
_seen: set[str] = set()
ALLOWED_HOSTS: list[str] = []
for _h in allowed_hosts:
    _h = (_h or '').strip()
    if not _h or _h in _seen:
        continue
    _seen.add(_h)
    ALLOWED_HOSTS.append(_h)

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'modules.dashboard',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'modules.dataanalyzer.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'modules.dataanalyzer.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Support Postgres (Railway) via DATABASE_URL si fourni
_database_url = os.environ.get('DATABASE_URL')
if _database_url:
    try:
        import dj_database_url

        DATABASES['default'] = dj_database_url.parse(
            _database_url,
            conn_max_age=600,
        )
    except Exception:
        # Fallback SQLite si parsing/driver indisponible
        pass

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'fr-fr'
TIME_ZONE = 'Europe/Paris'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

STORAGES = {
    'staticfiles': {
        'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage',
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Sessions en base SQLite (django.contrib.sessions)
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_NAME = 'dataanalyzer_sessionid'

# Dossiers projet
# Les datasets sont stockés dans modules/data/ dans ce repo
DATA_DIR = BASE_DIR / 'modules' / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
EXPORT_DIR = BASE_DIR / 'exports'

# Prod hardening (opt-in via env)
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    CSRF_TRUSTED_ORIGINS = [o.strip() for o in (os.environ.get('DJANGO_CSRF_TRUSTED_ORIGINS') or '').split(',') if o.strip()]
    SESSION_COOKIE_SECURE = _env_bool('DJANGO_SESSION_COOKIE_SECURE', default=True)
    CSRF_COOKIE_SECURE = _env_bool('DJANGO_CSRF_COOKIE_SECURE', default=True)
    SECURE_SSL_REDIRECT = _env_bool('DJANGO_SECURE_SSL_REDIRECT', default=False)
