
# Используем официальный образ Python
FROM python:3.8-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /mlflow

# Копируем файл зависимостей в рабочую директорию
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт 5000
EXPOSE 5000

# Запускаем MLflow server при старте контейнера
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "mysql://mlflow_user:mlflow_password@mysql:3306/mlflow", "--default-artifact-root", "s3://mlflow/"]

