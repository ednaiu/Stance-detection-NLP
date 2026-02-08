# Настройка SSH доступа для reg.ru

## 1. Получение SSH доступа на reg.ru

### Шаг 1: Получить данные для SSH подключения

1. Войдите в **Личный кабинет reg.ru**
2. Перейдите в раздел **"Хостинг и VPS"** или **"Серверы"**
3. Выберите ваш сервер/VPS (31.31.198.9)
4. Найдите раздел **"SSH доступ"** или **"Удаленный доступ"**

Там вы найдете:
- **IP адрес**: 31.31.198.9
- **SSH порт**: обычно 22 (но может быть другой)
- **Логин**: обычно `root` или имя вашего пользователя
- **Пароль**: предоставляется при заказе VPS или в панели управления

### Шаг 2: Создание SSH ключа (на вашем локальном компьютере)

#### Для Windows (PowerShell):

```powershell
# Открыть PowerShell и выполнить:
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Нажмите Enter для сохранения в папку по умолчанию:
# C:\Users\ednay\.ssh\id_rsa

# Введите пароль (или оставьте пустым для автоматического входа)
```

Будут созданы два файла:
- `C:\Users\ednay\.ssh\id_rsa` - **приватный ключ** (НИКОГДА не публикуйте!)
- `C:\Users\ednay\.ssh\id_rsa.pub` - **публичный ключ**

#### Для Linux/Mac:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# Файлы будут в ~/.ssh/id_rsa и ~/.ssh/id_rsa.pub
```

### Шаг 3: Копирование публичного ключа на сервер

#### Метод 1: Использование ssh-copy-id (Linux/Mac/WSL)

```bash
ssh-copy-id root@31.31.198.9
# Введите пароль от сервера
```

#### Метод 2: Вручную (Windows PowerShell)

```powershell
# 1. Скопировать содержимое публичного ключа
Get-Content C:\Users\ednay\.ssh\id_rsa.pub | Set-Clipboard

# 2. Подключиться к серверу по SSH
ssh root@31.31.198.9
# Введите пароль

# 3. На сервере выполнить:
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
# Вставить скопированный публичный ключ (правая кнопка мыши)
# Сохранить (Ctrl+O, Enter, Ctrl+X)

chmod 600 ~/.ssh/authorized_keys
exit
```

#### Метод 3: Через панель управления reg.ru

1. В панели управления reg.ru найдите раздел **"SSH ключи"**
2. Нажмите **"Добавить ключ"**
3. Скопируйте содержимое файла `id_rsa.pub`
4. Вставьте в форму и сохраните

### Шаг 4: Проверка подключения

```powershell
# Теперь должно подключаться БЕЗ пароля:
ssh root@31.31.198.9

# Если подключилось без запроса пароля - ключ работает!
```

## 2. Добавление SSH ключа в GitHub Secrets

### Получение приватного ключа для GitHub

```powershell
# Windows PowerShell:
Get-Content C:\Users\ednay\.ssh\id_rsa
```

```bash
# Linux/Mac:
cat ~/.ssh/id_rsa
```

Скопируйте ВСЁ содержимое (включая строки `-----BEGIN ... KEY-----` и `-----END ... KEY-----`)

Должно выглядеть так:
```
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn
...много строк...
-----END OPENSSH PRIVATE KEY-----
```

### Добавление в GitHub Secrets

1. Откройте ваш репозиторий на GitHub:
   https://github.com/ednaiu/Stance-detection

2. Перейдите в **Settings** (настройки репозитория)

3. В левом меню выберите **Secrets and variables** → **Actions**

4. Нажмите **New repository secret**

5. Создайте 3 секрета:

   **Секрет 1: REMOTE_HOST**
   - Name: `REMOTE_HOST`
   - Value: `31.31.198.9`

   **Секрет 2: REMOTE_USER**
   - Name: `REMOTE_USER`
   - Value: `root` (или ваш логин на сервере)

   **Секрет 3: SSH_PRIVATE_KEY**
   - Name: `SSH_PRIVATE_KEY`
   - Value: *[вставьте полное содержимое файла id_rsa]*

6. Нажмите **Add secret** для каждого

## 3. Настройка сервера (31.31.198.9)

После подключения к серверу по SSH выполните:

```bash
# 1. Установка Docker (если не установлен)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Добавление пользователя в группу docker
sudo usermod -aG docker $USER

# 3. Создание директорий для данных
mkdir -p /data/stance-detection/data
mkdir -p /data/stance-detection/models

# 4. Настройка firewall
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 5000/tcp # API
sudo ufw --force enable

# 5. Проверка Docker
docker --version
docker ps
```

## 4. Тестирование pipeline

### Локальный тест (без GitHub Actions)

```powershell
# Проверка SSH подключения
ssh root@31.31.198.9 "echo 'SSH работает!'"

# Запуск деплоя
.\deploy.ps1
```

### Автоматический деплой через GitHub Actions

После настройки секретов, просто выполните:

```powershell
.\commit_and_deploy.ps1 "Test deployment"
```

Или вручную:
```powershell
git add -A
git commit -m "Test deployment"
git push origin main
```

GitHub Actions автоматически задеплоит на сервер!

## Проверка статуса деплоя

### На GitHub:
1. Откройте ваш репозиторий
2. Перейдите во вкладку **Actions**
3. Увидите историю деплоев и логи

### На сервере:
```bash
ssh root@31.31.198.9

# Проверка контейнера
docker ps | grep stance-classifier

# Логи
docker logs stance-classifier

# Статистика
docker stats stance-classifier
```

### Проверка API:
```bash
curl http://31.31.198.9:5000/health
curl http://31.31.198.9:5000/
```

## Troubleshooting

### Проблема: "Permission denied (publickey)"

**Решение:**
```bash
# Проверьте, что ключ добавлен в SSH агент
ssh-add C:\Users\ednay\.ssh\id_rsa

# Или укажите ключ явно
ssh -i C:\Users\ednay\.ssh\id_rsa root@31.31.198.9
```

### Проблема: "Could not resolve hostname"

**Решение:**
```bash
# Проверьте доступность сервера
ping 31.31.198.9

# Проверьте SSH порт
telnet 31.31.198.9 22
```

### Проблема: Порт уже занят

**Решение:**
```bash
ssh root@31.31.198.9
# На сервере:
sudo lsof -i :5000
# Остановите процесс или смените порт
```

## Альтернатива: Использование пароля (не рекомендуется)

Если не хотите настраивать SSH ключ, можно использовать пароль:

```powershell
# Установить sshpass (только для тестирования!)
# НЕ используйте в продакшене!

$env:SSHPASS = "ваш_пароль"
sshpass -e ssh root@31.31.198.9
```

Но это **небезопасно** для production!

## Безопасность

✅ **Рекомендации:**
- Используйте SSH ключи вместо паролей
- Никогда не публикуйте приватный ключ
- Регулярно меняйте ключи
- Используйте разные ключи для разных проектов
- Включите двухфакторную аутентификацию на GitHub

❌ **Не делайте:**
- Не коммитьте приватные ключи в git
- Не используйте один ключ везде
- Не отключайте firewall
- Не запускайте Docker от root без необходимости
