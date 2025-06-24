# Gunakan image dasar Python yang kompatibel
FROM python:3.10-slim

# Atur working directory
WORKDIR /app

# Install dependensi sistem
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install lib untuk async Postgres
RUN pip install --upgrade pip

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh project ke dalam container
COPY . .

# Pastikan Chainlit bisa akses folder public
RUN ln -s /app/app/public /app/public && chmod -R 755 /app/app/public

# Expose port untuk Chainlit
EXPOSE 8000

# Jalankan Chainlit
CMD ["chainlit", "run", "app/app.py", "-w", "--host", "0.0.0.0", "--port", "8000"]

