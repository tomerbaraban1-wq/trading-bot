# 3.13 = the exact runtime the bot has run on locally for months (verified);
# the 3.11 image crashed on boot after the July 2026 feature push.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 8000

# PARKED: run the tiny stub instead of the full bot. The old cloud instance
# ("the ghost") was hijacking the owner's Telegram with stale answers, and the
# full app currently fails to boot on Render anyway. The stub replaces it with
# a harmless status page = equivalent to Suspend, but doable via git.
# To restore the real bot in the cloud: swap back to  uvicorn main:app .
CMD ["uvicorn", "cloud_stub:app", "--host", "0.0.0.0", "--port", "8000"]
