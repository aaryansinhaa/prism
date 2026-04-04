import asyncio
import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Body, File, HTTPException, UploadFile
from starlette import status

from runtime.model_loaders import load_model, default_model_path
from runtime.adapters.base import ModelPredictError, ModelLoadError


app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONTAINER_TEMPLATE_DIR = REPO_ROOT / "model_container"
ALLOWED_UPLOAD_SUFFIXES = {".onnx", ".pkl", ".pickle", ".joblib"}
DOCKER_BUILD_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_BUILD_TIMEOUT_SECONDS", "300"))
DOCKER_RUN_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_RUN_TIMEOUT_SECONDS", "60"))


def _upload_root() -> Path:
    configured = os.environ.get("MODEL_UPLOAD_ROOT")
    if configured:
        return Path(configured)
    return REPO_ROOT / "model_store" / "uploads"


def _save_uploaded_model(upload: UploadFile, model_dir: Path) -> Path:
    filename = Path(upload.filename or "model.bin").name
    target_path = model_dir / filename
    with target_path.open("wb") as target_file:
        shutil.copyfileobj(upload.file, target_file)
    return target_path


def _prepare_model_build_context(model_file_path: Path, model_dir: Path) -> Path:
    for template_file in ("runtime.py", "requirements.txt", "entrypoint.sh"):
        source_path = MODEL_CONTAINER_TEMPLATE_DIR / template_file
        destination_path = model_dir / template_file
        if not source_path.exists():
            raise RuntimeError(f"Missing container template file: {source_path}")
        shutil.copy2(source_path, destination_path)

    dockerfile_path = model_dir / "Dockerfile"
    dockerfile_contents = (
        "FROM python:3.13-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/requirements.txt\n"
        "RUN pip install --no-cache-dir -r /app/requirements.txt\n"
        "COPY runtime.py /app/runtime.py\n"
        "COPY entrypoint.sh /app/entrypoint.sh\n"
        f"COPY {model_file_path.name} /models/{model_file_path.name}\n"
        "RUN chmod +x /app/entrypoint.sh\n"
        f"ENV MODEL_PATH=/models/{model_file_path.name}\n"
        "ENV PORT=8000\n"
        "EXPOSE 8000\n"
        "ENTRYPOINT [\"/app/entrypoint.sh\"]\n"
    )
    dockerfile_path.write_text(dockerfile_contents, encoding="utf-8")
    return dockerfile_path


def _build_model_image(model_id: str, model_dir: Path) -> tuple[str, str]:
    image_tag = f"prism_model_{model_id}"
    command = ["docker", "build", "-t", image_tag, "."]
    try:
        result = subprocess.run(
            command,
            cwd=str(model_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=DOCKER_BUILD_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI not found. Please install Docker.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker build timed out after {DOCKER_BUILD_TIMEOUT_SECONDS}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or "docker build failed"
        raise RuntimeError(details) from exc

    return image_tag, (result.stdout or "").strip()


def _allocate_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_model_container(model_id: str, image_tag: str, host_port: int | None = None) -> tuple[str, int, str]:
    resolved_port = host_port or _allocate_host_port()
    container_name = f"prism_model_{model_id}"
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{resolved_port}:8000",
        image_tag,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=DOCKER_RUN_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI not found. Please install Docker.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker run timed out after {DOCKER_RUN_TIMEOUT_SECONDS}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or "docker run failed"
        raise RuntimeError(details) from exc

    container_id = (result.stdout or "").strip()
    return container_name, resolved_port, container_id


def _validate_upload_extension(file_name: str) -> None:
    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model format: {suffix}",
        )


async def _ingest_upload_and_build(file: UploadFile) -> Dict[str, Any]:
    model_id = uuid.uuid4().hex[:12]
    model_dir = _upload_root() / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_path = _save_uploaded_model(file, model_dir)
        dockerfile_path = _prepare_model_build_context(model_path, model_dir)
        image_tag, build_output = await asyncio.to_thread(_build_model_image, model_id, model_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    finally:
        await file.close()

    return {
        "model_id": model_id,
        "image_tag": image_tag,
        "model_path": str(model_path),
        "dockerfile_path": str(dockerfile_path),
        "build_context": str(model_dir),
        "build_output": build_output,
    }


@app.on_event("startup")
async def load_runtime_model() -> None:
    """Load model at application startup and attach to app.state.model.

    The loader uses `MODEL_PATH` env var if set, otherwise falls back to
    `model_store/linear_regression.onnx` in the repository.
    """
    model_path = os.environ.get("MODEL_PATH") or default_model_path()
    try:
        model = load_model(model_path)
        app.state.model = model
        print(f"Loaded model from: {model_path}")
    except Exception as exc:  # keep startup resilient and surface load errors
        # If the real model cannot be loaded (missing optional deps, etc.)
        # install a small in-memory fallback model so the service can be
        # exercised during development.
        print(f"Failed to load model from {model_path}: {exc}")

        class _FallbackModel:
            def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Echo input to allow quick smoke-testing via curl.
                return {"predictions": input_data}

        app.state.model = _FallbackModel()
        print("Registered fallback model for development/testing.")


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"service": "prism-runtime"}


@app.post("/predict")
async def predict(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run prediction using the loaded model.

    This endpoint accepts a JSON mapping in the body and returns a
    JSON-serializable response produced by the adapter.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        normalized_payload = getattr(model, "validate_input", lambda value: value)(payload)
        result = model.predict(normalized_payload)
    except ModelPredictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return result


@app.post("/upload")
async def upload_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_name = file.filename or ""
    _validate_upload_extension(file_name)

    return await _ingest_upload_and_build(file)


@app.post("/upload-and-run")
async def upload_and_run_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_name = file.filename or ""
    _validate_upload_extension(file_name)

    upload_result = await _ingest_upload_and_build(file)

    model_id = upload_result["model_id"]
    image_tag = upload_result["image_tag"]

    try:
        container_name, host_port, container_id = await asyncio.to_thread(
            _run_model_container,
            model_id,
            image_tag,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    upload_result.update(
        {
            "container_name": container_name,
            "container_id": container_id,
            "host_port": host_port,
            "predict_url": f"http://127.0.0.1:{host_port}/predict",
        }
    )
    return upload_result
