
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run training to generate the model
 RUN python train.py

# Expose the port Cloud Run expects
EXPOSE 8080

# Override default Flask port (5000 → 8080) via env + args
ENV FLASK_APP=app.py
ENV PORT=8080

# Run Flask on 0.0.0.0:8080 without modifying app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
