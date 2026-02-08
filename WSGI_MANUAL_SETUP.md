# WSGI Deployment - Manual Setup for Shared Hosting

Так как git clone не работает с GitHub на хостинге, используйте этот способ:

## Вариант 1: Загрузить архив через FTP (Рекомендуемый)

### Шаг 1: Скачать архив локально
```bash
# На вашем компьютере
git clone https://github.com/ednaiu/Stance-detection.git
cd Stance-detection
git archive --format zip --output stance-detection.zip HEAD
```

### Шаг 2: Загрузить через FTP
- Откройте панель управления https://server232.hosting.reg.ru:1500/
- Перейдите в File Manager
- Навигируйте в `/data/`
- Загрузите `stance-detection.zip`
- Распакуйте архив

### Шаг 3: Установить зависимости по SSH
```bash
ssh u3089870@31.31.198.9
cd /var/www/u3089870/data/stance-detection
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r StanceClassifier/requirements.txt
```

## Вариант 2: Скопировать файлы через SCP

```bash
# На вашем компьютере
cd c:\Users\ednay\PycharmProjects\stance_detection

# Загрузить проект
scp -r . u3089870@31.31.198.9:/var/www/u3089870/data/stance-detection

# Установить зависимости
ssh u3089870@31.31.198.9 << 'EOF'
cd /var/www/u3089870/data/stance-detection
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r StanceClassifier/requirements.txt
EOF
```

## Вариант 3: Прямое подключение и создание файлов

```bash
ssh u3089870@31.31.198.9 << 'EOF'
cd /var/www/u3089870/data

# Создать структуру вручную
mkdir -p stance-detection/StanceClassifier
mkdir -p stance-detection/models
mkdir -p stance-detection/data

# Содержимое wsgi.py можно загрузить в файл или скопировать вручную
cat > stance-detection/wsgi.py << 'PYEOF'
# ... содержимое файла wsgi.py ...
PYEOF

# ... и т.д. для других файлов

# Затем создать виртуальное окружение
cd stance-detection
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r StanceClassifier/requirements.txt
EOF
```

## Рекомендуемый процесс

1. **Используйте Вариант 1 (FTP)** - самый простой если у вас есть GUI
2. **Или Вариант 2 (SCP)** - если удобнее через командную строку

Выполните выбранный вариант и отчитайтесь о результатах!

## Проверка после установки

```bash
ssh u3089870@31.31.198.9
cd /var/www/u3089870/data/stance-detection
source venv/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

Если вывод показывает версии - значит всё установилось корректно!

## Запуск приложения

```bash
source venv/bin/activate
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 60 wsgi:app

# Или в фоне через screen
screen -S stance
./run.sh
```

Затем проверите http://31.31.198.9:5000/health в браузере.
