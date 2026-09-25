FROM python:3.13-slim
WORKDIR /app
COPY backend backend
RUN pip install --no-cache-dir -e './backend[dev,postgres,speech,visual]'
COPY tests tests
COPY ml ml
COPY scripts scripts
RUN useradd --create-home scenemind && mkdir -p /app/data && chown scenemind /app/data
USER scenemind
ENV SCENEMIND_DURABLE_JOBS=true
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
