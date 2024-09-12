# Usa una imagen base de Python 3.8
FROM python:3.8-slim

# Instala las dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto a la imagen
COPY . .

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto de Flask
EXPOSE 5000

# Comando para correr la aplicación
CMD ["python", "app.py"]