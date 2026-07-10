# 3.13 = the exact runtime the bot has run on locally for months (verified);
# the 3.11 image crashed on boot after the July 2026 feature push.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
