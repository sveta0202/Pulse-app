# PULSE: ЭМГ → движения кисти

Один сервис: FastAPI отдаёт готовый интерфейс PULSE и принимает исходную запись ЭМГ на `POST /api/predict`. Модель один раз загружается в память при старте. На запросе нет обучения, конкурсного шаблона или обращения к датасету.

**Для настоящих прогнозов нужен `weights/pulse.pt`. В исходной поставке этого файла нет.** В приложенном `lite.py` чекпоинты `work_lite/.../*.pkl` содержат готовые вероятности или предсказания для тех записей, на которых запускали скрипт, а не обученные сети для новых файлов. Чтобы получить `pulse.pt`, запустите экспорт на машине, где есть разметка AIRI, либо попросите ML-участника команды выполнить эту команду. Если у него сохранены настоящие веса в другом формате, экспорт из них потребует точного формата и нормировок.

## Структура репозитория

```text
pulse-railway/
├── .railway/railway.ts      # IaC: источник GitHub и healthcheck
├── .dockerignore
├── .gitignore
├── Dockerfile               # Python 3.12 + PyTorch CPU
├── package.json             # Railway IaC SDK, не нужен Python-сервису
├── package-lock.json
├── requirements.txt
├── requirements-dev.txt
├── server.py                # HTTP, лимиты, готовность, HTML и примеры
├── model_adapter.py         # проверка входа и инференс
├── export_model.py          # офлайн-обучение и экспорт весов
├── lite.py                  # архитектуры и обучение; конкурсный CLI сохранён
├── common.py
├── feats.py
├── clf_features.py
├── static/
│   ├── index.html           # фронтенд PULSE
│   └── examples/           # пять записей для лаборатории
├── weights/
│   ├── .gitkeep
│   └── pulse.pt             # создаётся export_model.py, отсутствует в архиве
├── tests/test_integration.py
├── LITE.md
├── ML_INTEGRATION.md
└── README.md
```

## Экспорт весов

На Linux x86-64 с Python 3.12 из корня репозитория:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python export_model.py --data /absolute/path/to/AIRI --threads 2
ls -lh weights/pulse.pt
```

`--data` указывает на каталог, содержащий `train/metadata.csv`, `train/signals/<recording_id>.npy` и `train/targets/<recording_id>.npy`. В `metadata.csv` нужны `recording_id` и `action`; обучение ожидает все классы из `common.ACTIONS`. Нужны исходные сигналы AIRI в том же порядке и масштабе каналов, что и при обучении.

По умолчанию регрессор с архитектурой студента обучается на размеченном `train` без дистилляции. Если хотите сохранить подход «учитель → студент» из `lite.py`, добавьте `--distill-from /absolute/path/to/unlabeled-signals` с неразмеченными `.npy` из подходящего домена:

```bash
python export_model.py --data /absolute/path/to/AIRI \
  --distill-from /absolute/path/to/unlabeled-signals --threads 2
```

Веб-версия выбирает движение по максимуму ансамбля двух CNN и LogReg. Венгерское распределение подходит для специально сбалансированного конкурсного батча, но не для одного произвольного файла. Экспорт включает параметры сетей, коэффициенты LogReg и нормировки; учителя нужны только при офлайн-дистилляции. Это веб-версия пайплайна, а не побитовое воспроизведение конкурсных предсказаний. Для честной оценки на новом дне не включайте его записи в дистилляцию.

## Локальный запуск

```bash
python server.py
```

Откройте `http://localhost:8000`, выберите `.npy` или `.csv` и нажмите «Обработать сигнал». Не открывайте HTML через `file://`: браузеру нужен HTTP-сервер на том же домене, что и `/api/predict`.

Можно также нажать «Запись 01» … «Запись 05» в блоке «Готовые примеры ЭМГ». Сайт возьмёт соответствующий `.npy` из `static/examples/` и сразу запустит тот же инференс. Эти файлы доступны посетителям через `/examples/01` … `/examples/05`; перед публикацией проверьте, что записи разрешено показывать. Файлы пользовательской загрузки по этим адресам недоступны.

Контрольный запрос:

```bash
BASE=http://localhost:8000
curl -fsS "$BASE/healthz"
curl -fsS "$BASE/api/predict" \
  -H 'Content-Type: application/octet-stream' \
  -H 'X-Filename: recording.npy' \
  --data-binary @/absolute/path/to/recording.npy \
  -o angles.csv
head -n 2 angles.csv
```

В Docker:

```bash
docker build -t pulse:local .
docker run --rm -p 8000:8000 -e PORT=8000 pulse:local
```

## GitHub → Railway

1. Создайте пустой приватный репозиторий. Скопируйте в его корень содержимое каталога `pulse-railway/`, включая скрытые файлы, а не сам каталог. Добавьте проверенный `weights/pulse.pt`, только если правила хакатона и команды позволяют хранить веса в GitHub; иначе этот вариант поставки весов замените приватным хранилищем, а `MODEL_PATH` направьте на подготовленный файл. `data/`, исходные записи ЭМГ и токены не коммитьте.
2. В `.railway/railway.ts` замените `YOUR_GITHUB_OWNER/YOUR_GITHUB_REPO` на путь вашего репозитория, при необходимости измените ветку. В корне файлов выполните:

   ```bash
   git init -b main
   git add .
   git status --short
   git commit -m "Add PULSE inference service"
   git remote add origin git@github.com:YOUR_GITHUB_OWNER/YOUR_GITHUB_REPO.git
   git push -u origin main
   ```

3. **Быстрый путь через интерфейс.** На Railway выберите `New Project → Deploy from GitHub repo`, подключите GitHub App к нужному репозиторию и выберите `main`. Оставьте корень `/` и не задавайте Start Command: используется `CMD` из корневого `Dockerfile`. Проверьте в build logs, что Dockerfile обнаружен. В `Settings → Deploy` задайте `Healthcheck Path = /healthz` и timeout `120`; в переменных оставьте `PORT` под управлением Railway, `MODEL_THREADS` по умолчанию равен `1`. После сохранения примените изменения и дождитесь готовности. См. [Railway Quick Start](https://docs.railway.com/quick-start), [Dockerfiles](https://docs.railway.com/guides/dockerfiles) и [Healthchecks](https://docs.railway.com/guides/healthchecks).
4. **Альтернатива с конфигурацией в коде.** Если хотите управлять сервисом из репозитория, не создавайте второй сервис из шага 3. Создайте отдельный пустой Railway-проект, подключите GitHub к аккаунту и дайте Railway App доступ к репозиторию. Нужны свежий Railway CLI с `config` и Node.js 22+ для SDK. Из корня:

   ```bash
   npm ci
   railway login
   railway link
   railway config plan
   railway config apply
   ```

   При `railway link` выберите выделенный проект и нужное окружение. Перед `apply` проверьте, что план создаёт только сервис `pulse` и не удаляет ресурсы. `.railway/railway.ts` применяется CLI, а не считывается при каждом GitHub-деплое. `railway.toml` / `railway.json` не используются: для новых сервисов Railway их больше не предлагает ([Railway IaC](https://docs.railway.com/infrastructure-as-code), [устаревший Config as Code](https://docs.railway.com/guides/config-as-code)).
5. Когда в логах есть `PULSE model loaded` и `/healthz` проходит, откройте `Settings → Networking → Public Networking → Generate Domain`. Для проверки задайте `BASE=https://<выданный-домен>` и повторите запросы `curl`, затем обработайте известную запись через браузер ([Railway Quick Start](https://docs.railway.com/quick-start)).

Сервис рассчитан на хакатонное демо. Публичный домен открывает API без аутентификации; для длительного публичного использования добавьте контроль доступа и ограничения на внешнем прокси. Сервер обрабатывает данные запроса в памяти; не публикуйте записи участников без разрешения.

## HTTP-контракт и диагностика

- **Вход:** сырое тело, `Content-Type: application/octet-stream` и заголовок `X-Filename`. Поддерживаются little-endian C-order `.npy` с `float32`/`float64` и UTF-8 `.csv` с запятой или точкой с запятой, десятичной точкой и необязательным заголовком из шести идентификаторов. Частота `500 Гц`, форма `(T,6)`. Допустимы `5 000–60 000` отсчётов на канал (примерно `10–120` секунд); ограничение тела `12 МиБ`. Нижний порог задаёт окно классификатора; частоту из самих значений сервер определить не может.
- **Выход:** `200`, `text/csv; charset=utf-8`, по строке на входной отсчёт, `angle_00` … `angle_19` в радианах. Для сайта возвращается вся запись. Это не конкурсный submission: официальный шаблон с идентификаторами и отсечением контекста по-прежнему обслуживает `lite.py predict`.
- **Ошибки:** `422` для неверного входа; `413` для слишком большого тела; `415` для неверного типа; `429` при занятом слоте; `501` если файла весов нет; `500` для повреждённых весов или ошибки выполнения. Если диагностика показывает `/healthz → 503`, проверьте наличие весов и стартовые логи.
- **Параметры:** `PORT` по умолчанию `8000`; `MODEL_THREADS` по умолчанию `1`; `MODEL_PATH` по умолчанию `weights/pulse.pt` относительно кода сервера. Загружайте только экспортированные командой веса из доверенного источника.

## Проверки

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Тесты временно создают синтетическую ЭМГ, выполняют сокращённое обучение и экспорт, затем вызывают API с локально загруженной моделью. Эти веса не входят в релиз. Тесты проверяют интеграцию и формат, но не точность на реальной ЭМГ.
