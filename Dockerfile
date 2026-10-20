FROM python:3.10-slim

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

# Run train_model.py to generate the model at build time
RUN python train_model.py

# Expose HuggingFace Spaces port
EXPOSE 7860

# Start the Flask app on port 7860
CMD ["python", "-m", "gunicorn", "--chdir", "app", "--bind", "0.0.0.0:7860", "main:app"]
