# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY artifacts/ artifacts/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
COPY --from=frontend-build /app/frontend/dist frontend/dist

RUN mkdir -p /app/data

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
