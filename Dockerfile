# Dockerfile 

# 1. Base
FROM python:3.9-slim

# 2. Work directory
WORKDIR /app

# 3. Install the dependencies
RUN apt-get update && apt-get install -y build-essential

# 4. Copy dependencies
COPY requirements.txt .

# 5. Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# 7. Port
EXPOSE 8000

# 8. Run command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]