# Безопасное хранение паролей и credentials

## ⚠️ ВАЖНО: Не храните пароли в открытом виде!

## Вариант 1: Файл .env (Рекомендуется для разработки)

### Создайте файл .env

В корне проекта создайте файл `.env`:

```bash
REMOTE_HOST=31.31.198.9
REMOTE_USER=root
REMOTE_PASSWORD=ваш_реальный_пароль
```

### Добавьте .env в .gitignore

**Обязательно** убедитесь, что `.env` в `.gitignore`:

```bash
# В файле .gitignore должна быть строка:
.env
.env.*
!.env.example
```

### Загрузите переменные из .env

```powershell
# Создайте скрипт load_env.ps1:
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.+)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        Set-Item -Path "env:$name" -Value $value
    }
}
```

Использование:
```powershell
# Загрузить переменные
. .\load_env.ps1

# Теперь можно деплоить
.\deploy.ps1
```

## Вариант 2: Windows Credential Manager (Самый безопасный для Windows)

### Сохранение пароля в Credential Manager

```powershell
# Сохранить пароль (выполнить один раз)
$credential = Get-Credential -UserName "root@31.31.198.9" -Message "Введите пароль для сервера"

# Сохранить в Windows Credential Manager
$credential.Password | ConvertFrom-SecureString | Out-File "$env:USERPROFILE\.ssh\server_password.txt"
```

### Чтение пароля из Credential Manager

Создайте файл `load_credentials.ps1`:

```powershell
# load_credentials.ps1
$passwordFile = "$env:USERPROFILE\.ssh\server_password.txt"

if (Test-Path $passwordFile) {
    $securePassword = Get-Content $passwordFile | ConvertTo-SecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    $env:REMOTE_PASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    $env:REMOTE_USER = "root"
    $env:REMOTE_HOST = "31.31.198.9"
    
    Write-Host "Credentials loaded successfully" -ForegroundColor Green
} else {
    Write-Host "Password file not found. Please run setup first." -ForegroundColor Red
}
```

Использование:
```powershell
# Первый раз сохранить пароль
$credential = Get-Credential -UserName "root" -Message "Введите пароль"
$credential.Password | ConvertFrom-SecureString | Out-File "$env:USERPROFILE\.ssh\server_password.txt"

# Потом каждый раз загружать
. .\load_credentials.ps1
.\deploy.ps1
```

## Вариант 3: PowerShell Profile (Постоянные переменные)

### Создание зашифрованного профиля

```powershell
# Откройте профиль PowerShell
notepad $PROFILE

# Добавьте в профиль:
function Load-DeploymentCredentials {
    $passwordFile = "$env:USERPROFILE\.ssh\server_password.txt"
    
    if (Test-Path $passwordFile) {
        $securePassword = Get-Content $passwordFile | ConvertTo-SecureString
        $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
        $env:REMOTE_PASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
        $env:REMOTE_USER = "root"
        $env:REMOTE_HOST = "31.31.198.9"
        
        Write-Host "✓ Deployment credentials loaded" -ForegroundColor Green
    }
}

# Автоматически загружать при запуске PowerShell (опционально)
# Load-DeploymentCredentials
```

Сохраните и перезапустите PowerShell. Теперь в любой момент можно вызвать:

```powershell
Load-DeploymentCredentials
.\deploy.ps1
```

## Вариант 4: GitHub Secrets (Для CI/CD)

Для автоматического деплоя через GitHub Actions:

1. **Никогда** не храните пароли в коде
2. Используйте **GitHub Secrets**:
   - Откройте: https://github.com/ednaiu/Stance-detection/settings/secrets/actions
   - Добавьте секреты:
     - `REMOTE_HOST` = `31.31.198.9`
     - `REMOTE_USER` = `root`
     - `REMOTE_PASSWORD` = `ваш_пароль`

3. Секреты **зашифрованы** и доступны только в GitHub Actions

## Вариант 5: Интерактивный ввод (Самый простой и безопасный)

Модифицируйте `deploy.ps1` для запроса пароля:

```powershell
# В начале скрипта deploy.ps1:
if (-not $env:REMOTE_PASSWORD) {
    $securePassword = Read-Host "Введите пароль для root@31.31.198.9" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    $env:REMOTE_PASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
}
```

Теперь просто запускайте:
```powershell
.\deploy.ps1
# Введите пароль: ********
```

Пароль **не сохраняется** нигде!

## Создание файла .env.example (для других разработчиков)

Создайте `.env.example` с шаблоном (без реальных паролей):

```bash
# .env.example - скопируйте в .env и заполните реальными значениями
REMOTE_HOST=31.31.198.9
REMOTE_USER=root
REMOTE_PASSWORD=your_password_here
```

Этот файл **можно** коммитить в git.

## Проверка безопасности

### ✅ Что НУЖНО делать:

```powershell
# Проверить, что .env в .gitignore
git check-ignore .env
# Должно вывести: .env

# Убедиться, что пароль не в истории git
git log --all --full-history --source -- '*password*' '*env*'

# Проверить, что переменные загружены
echo $env:REMOTE_PASSWORD
# Должен показать пароль (только для проверки!)
```

### ❌ Что НЕ НУЖНО делать:

- ❌ Не добавляйте пароли в код
- ❌ Не коммитьте файл `.env`
- ❌ Не храните пароли в переменных PowerShell постоянно
- ❌ Не отправляйте пароли в чаты/email
- ❌ Не используйте слабые пароли

## Рекомендуемая стратегия

### Для локальной разработки:

**Вариант A (простой):**
```powershell
# Каждый раз вводить пароль
.\deploy.ps1
# Введите пароль: ********
```

**Вариант B (удобный):**
```powershell
# Один раз настроить
. .\load_env.ps1  # загружает из .env файла

# Или
. .\load_credentials.ps1  # загружает из зашифрованного хранилища

# Потом деплоить
.\deploy.ps1
```

### Для автоматического деплоя:

Используйте **GitHub Secrets** - они зашифрованы и безопасны.

## Дополнительная безопасность

### 1. Ограничить доступ к .env файлу

```powershell
# Сделать файл доступным только вам
$acl = Get-Acl .env
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $env:USERNAME, "FullControl", "Allow"
)
$acl.SetAccessRule($rule)
Set-Acl .env $acl
```

### 2. Использовать разные пароли

Не используйте один и тот же пароль для:
- reg.ru панели управления
- SSH доступа к серверу
- GitHub аккаунта
- Других сервисов

### 3. Регулярно менять пароли

Меняйте пароли каждые 3-6 месяцев.

### 4. Включить двухфакторную аутентификацию

На всех сервисах, где возможно (GitHub, reg.ru и т.д.)

## Быстрая настройка (Рекомендуемая)

Скопируйте и выполните:

```powershell
# 1. Создать .env файл
@"
REMOTE_HOST=31.31.198.9
REMOTE_USER=root
REMOTE_PASSWORD=ваш_реальный_пароль_здесь
"@ | Out-File -FilePath .env -Encoding UTF8

# 2. Проверить, что .env в .gitignore
if (-not (Get-Content .gitignore | Select-String "^\.env$")) {
    Add-Content .gitignore "`n.env"
}

# 3. Создать скрипт загрузки
@"
# Load environment variables from .env file
Get-Content .env | ForEach-Object {
    if (`$_ -match '^([^=]+)=(.+)`$') {
        `$name = `$matches[1].Trim()
        `$value = `$matches[2].Trim()
        Set-Item -Path "env:`$name" -Value `$value
    }
}
Write-Host "Environment variables loaded from .env" -ForegroundColor Green
"@ | Out-File -FilePath load_env.ps1 -Encoding UTF8

Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host "Usage: . .\load_env.ps1 ; .\deploy.ps1" -ForegroundColor Cyan
```

Теперь используйте:
```powershell
. .\load_env.ps1
.\deploy.ps1
```

## Troubleshooting

### Проблема: "Пароль не загружается"

```powershell
# Проверить, что файл существует
Test-Path .env

# Проверить содержимое (ОСТОРОЖНО - покажет пароль!)
Get-Content .env

# Проверить переменные окружения
Get-ChildItem env: | Where-Object { $_.Name -like "REMOTE_*" }
```

### Проблема: "Git shows .env in changes"

```bash
# Удалить из индекса (но оставить локально)
git rm --cached .env

# Добавить в .gitignore
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to gitignore"
```

## Итоговая рекомендация

**Для начинающих:**
- Используйте интерактивный ввод пароля
- Скрипт сам спросит пароль при запуске

**Для продвинутых:**
- Используйте `.env` файл + `load_env.ps1`
- Не забудьте добавить `.env` в `.gitignore`

**Для production:**
- Используйте GitHub Secrets для CI/CD
- Используйте SSH ключи вместо паролей (см. SSH_SETUP_REGRU.md)
