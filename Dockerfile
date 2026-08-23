# python:3.12-slim is glibc-based, so the manylinux wheels that fail on NixOS install
# normally here. This is the path of least resistance on any host.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so edits to the source do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root. The app only reads the repo and writes .cache/, which it owns.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/.cache \
    && chown -R app:app /app
USER app

EXPOSE 8994

# 0.0.0.0 so the port is reachable from the host; publish it with -p 8994:8994.
CMD ["python", "-m", "uvicorn", "agenttrace.api:app", \
     "--host", "0.0.0.0", "--port", "8994"]
