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

# The container always listens on 8000. A container port lives in its own namespace and so
# can never collide with anything on the host; only the published host port can. Change the
# host side (compose PORT, or -p) and leave this alone.
ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} is expanded; exec so uvicorn is PID 1 and receives signals.
CMD ["sh", "-c", "exec python -m uvicorn agenttrace.api:app --host 0.0.0.0 --port ${PORT}"]
