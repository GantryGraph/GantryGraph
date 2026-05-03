FROM python:3.11-slim

# Xvfb provides a virtual display so pyautogui/desktop tools work headlessly on Linux
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

# Install with cloud + desktop extras
RUN pip install --no-cache-dir -e ".[cloud,desktop]"

# Optional: install playwright browsers
# RUN playwright install chromium

# Set virtual display for headless desktop automation
ENV DISPLAY=:99

# Start Xvfb virtual display then launch the agent API server
CMD ["sh", "-c", \
  "Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &>/dev/null & \
   exec python -c 'from claw.cloud.serve import serve; serve(lambda: None)'"]
