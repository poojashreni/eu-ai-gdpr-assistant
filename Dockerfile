# Hugging Face Space (Docker SDK) image: FastAPI on 127.0.0.1:8000, Streamlit on 0.0.0.0:7860.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# Hugging Face Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    HF_HOME=/home/user/app/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    API_URL=http://127.0.0.1:8000
RUN mkdir -p /home/user/app/data
WORKDIR /home/user/app

# Dependencies first (cached layer). The Linux lock entry for torch is the CPU wheel.
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev

COPY --chown=user app ./app
COPY --chown=user data/raw ./data/raw
COPY --chown=user start.sh ./start.sh

# Build the chunks and the index now, so the e5 model and the index are baked into the image
# and the Space starts without downloading or embedding anything.
RUN python -m assistant.parse && python -m assistant.index

EXPOSE 7860
CMD ["bash", "start.sh"]
