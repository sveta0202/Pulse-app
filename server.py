import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from model_adapter import InputError, PulseModel, predict_emg_file

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 12 * 1024**2
LOGGER = logging.getLogger("uvicorn.error")


def create_app(weights=None):
    weights = Path(weights if weights is not None else os.getenv("MODEL_PATH", ROOT / "weights/pulse.pt"))

    @asynccontextmanager
    async def lifespan(app):
        import torch

        torch.set_num_threads(max(1, int(os.getenv("MODEL_THREADS", "1"))))
        app.state.model = None
        app.state.missing = False
        app.state.slot = asyncio.Lock()
        try:
            app.state.model = await run_in_threadpool(PulseModel, weights)
            LOGGER.info("PULSE model loaded: %s", weights)
        except FileNotFoundError:
            app.state.missing = True
            LOGGER.warning("PULSE checkpoint not found: %s", weights)
        except Exception:
            LOGGER.exception("PULSE checkpoint could not be loaded")
        yield

    app = FastAPI(title="PULSE", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "static/index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/healthz")
    async def health():
        ready = app.state.model is not None
        return JSONResponse({"status": "ready" if ready else "not_ready"}, status_code=200 if ready else 503)

    @app.post("/api/predict")
    async def predict(request: Request):
        if app.state.model is None:
            if app.state.missing:
                return PlainTextResponse("Веса модели не установлены.", status_code=501)
            return PlainTextResponse("Не удалось загрузить модель.", status_code=500)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/octet-stream":
            return PlainTextResponse("Ожидается application/octet-stream.", status_code=415)
        try:
            filename = unquote(request.headers.get("x-filename", ""), errors="strict")
        except UnicodeError:
            return PlainTextResponse("Некорректный X-Filename.", status_code=422)
        if Path(filename).suffix.lower() not in {".npy", ".csv"}:
            return PlainTextResponse("X-Filename должен указывать файл .npy или .csv.", status_code=422)
        length = request.headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                return PlainTextResponse("Некорректный Content-Length.", status_code=422)
            if size > MAX_BYTES:
                return PlainTextResponse("Лимит файла: 12 МиБ.", status_code=413)
        if app.state.slot.locked():
            return PlainTextResponse("Модель занята. Повторите запрос.",
                                     status_code=429, headers={"Retry-After": "2"})
        async with app.state.slot:
            body = bytearray()
            try:
                async with asyncio.timeout(30):
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_BYTES:
                            return PlainTextResponse("Лимит файла: 12 МиБ.", status_code=413)
                        body.extend(chunk)
                result = await run_in_threadpool(predict_emg_file, app.state.model, body, filename)
            except (TimeoutError, ClientDisconnect):
                return PlainTextResponse("Загрузка файла не завершилась.", status_code=408)
            except InputError as exc:
                return PlainTextResponse(str(exc), status_code=422)
            except Exception:
                LOGGER.exception("PULSE inference failed")
                return PlainTextResponse("Внутренняя ошибка инференса.", status_code=500)
        return Response(result, media_type="text/csv", headers={"Cache-Control": "no-store"})

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
