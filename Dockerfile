# UNTESTED: written but not built (Docker's service was not running when this was prepared). Code + dependencies only: mount data/ as a volume.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000 DEVICE=cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY web ./web
COPY run.sh .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["./run.sh"]
