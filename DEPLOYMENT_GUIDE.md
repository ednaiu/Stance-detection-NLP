# Stance Detection - Deployment Options

Данный проект поддерживает несколько способов развертывания:

## 🐳 Вариант 1: Docker (Рекомендуемый)

**Для**: VPS, облачные серверы (AWS, DigitalOcean, Hetzner), локальная разработка

**Требования**: Docker и Docker Compose

```bash
cd StanceClassifier
docker-compose -f docker/docker-compose.yml up -d
```

**Плюсы**:
- ✅ Полная изоляция окружения
- ✅ Легко масштабировать
- ✅ Стандартный процесс развертывания
- ✅ CI/CD интеграция

**Минусы**:
- ❌ Требует прав sudo/администратора
- ❌ Больше ресурсов (памяти и CPU)

**Документация**: [Docker Setup](StanceClassifier/docker/README.md)

---

## 🐍 Вариант 2: WSGI (Для Shared Hosting)

**Для**: reg.ru shared hosting, другие традиционные хостинги

**Требования**: Python 3.6+, pip, venv, git

### Быстрое начало:

```bash
# Для Linux/Mac/WSL
bash deploy_wsgi.sh u3089870 31.31.198.9

# Для Windows (через Git Bash)
bash deploy_wsgi.sh u3089870 31.31.198.9

# Или вручную запустить PowerShell скрипт
.\deploy_wsgi.ps1 -User u3089870 -Host 31.31.198.9
```

### Ручное развертывание:

```bash
ssh u3089870@31.31.198.9
cd /var/www/u3089870/data/stance-detection
git clone https://github.com/ednaiu/Stance-detection.git .

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate
pip install -r StanceClassifier/requirements.txt

# Запустить
gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app
```

**Плюсы**:
- ✅ Работает на любом хостинге с Python
- ✅ Меньше ресурсов
- ✅ Легко управлять через screen/nohup
- ✅ Прямой доступ к приложению

**Минусы**:
- ❌ Нужно ручное управление процессом
- ❌ Сложнее масштабировать
- ❌ Нет изоляции окружения

**Документация**: [WSGI Deployment](WSGI_DEPLOYMENT.md)

---

## 🚀 Вариант 3: Локально на Windows

**Для**: Разработка, тестирование

```bash
cd StanceClassifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Запустить Flask приложение
python -m flask --app ..\wsgi:app run
```

**Документация**: [Local Development](StanceClassifier/README.md)

---

## 📊 Сравнение вариантов

| Параметр | Docker | WSGI | Локально |
|----------|--------|------|----------|
| Легкость | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Надежность | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Масштабируемость | ⭐⭐⭐ | ⭐ | ⭐ |
| Ресурсы | ❌ много | ✅ мало | ✅ мало |
| Production | ✅ ДА | ⚠️ ДА | ❌ НЕТ |

---

## 🔄 CI/CD Pipeline

Для автоматического развертывания используется GitHub Actions.

**Конфигурация**: [.github/workflows/deploy.yml](.github/workflows/deploy.yml)

### Для Docker (VPS):

```bash
# Подготовить GitHub Secrets:
# - REMOTE_HOST=your.server.ip
# - REMOTE_USER=root
# - REMOTE_PASSWORD=your_password

# Затем просто push в main
git push origin main

# GitHub Actions автоматически:
# 1. Соберет Docker образ
# 2. Загрузит на сервер
# 3. Запустит контейнер
```

### Для WSGI (Shared Hosting):

```bash
# Обновить скрипт развертывания для WSGI
# Или использовать webhook для git pull
```

---

## 📋 Выбор варианта

**Используйте Docker если**:
- ✅ У вас есть VPS или облачный сервер
- ✅ Нужна высокая надежность
- ✅ Хотите автоматизировать развертывание
- ✅ Планируете масштабировать приложение

**Используйте WSGI если**:
- ✅ Используете shared hosting (reg.ru, etc)
- ✅ Нет возможности использовать Docker
- ✅ Нужна простая настройка
- ✅ Низкие требования по ресурсам

**Используйте локально если**:
- ✅ Разрабатываете на Windows/Mac
- ✅ Тестируете новые функции
- ✅ Не нужен production сервер

---

## 🆘 Решение проблем

### Docker не работает

```bash
# Проверить статус Docker
docker ps

# Если не запущен
sudo systemctl start docker

# Пересоздать контейнер
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml up -d
```

### WSGI приложение медленное

```bash
# Увеличить количество рабочих процессов
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

# Или через скрипт
sed -i 's/-w 2/-w 4/g' run.sh
./run.sh
```

### Не загружаются модели

```bash
# Скачать модели заранее
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    'strombergnlp/rumoureval_2019',
    repo_type='model',
    local_dir='models/sentence_embedding_baseline'
)
EOF
```

---

## 📚 Документация

- [Docker Development](StanceClassifier/docker/README.md)
- [WSGI Deployment](WSGI_DEPLOYMENT.md)
- [Server Setup Checklist](SERVER_SETUP_CHECKLIST.md)
- [API Reference](StanceClassifier/README.md)

---

## 🔗 Полезные ссылки

- [GitHub Repository](https://github.com/ednaiu/Stance-detection)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [Docker Documentation](https://docs.docker.com/)

---

**Версия**: 2.5.1
**Последнее обновление**: 2026-02-08
