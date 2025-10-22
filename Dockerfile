# 1. Base Image: Use a specific, lightweight Python version.
# python:3.11-slim is a great choice for production as it's smaller.
FROM python:3.11-slim

# 2. Set the working directory inside the container.
# All subsequent commands (COPY, RUN, CMD) will be relative to this path.
WORKDIR /app

# 3. Copy dependencies file first to leverage Docker's build cache.
# This layer is only rebuilt if your requirements.txt file changes.
COPY requirements_version.txt .

# 4. Install the dependencies.
# --no-cache-dir keeps the final image size smaller.
RUN pip install --no-cache-dir -r requirements_version.txt

# 5. Copy your application code into the container.
# The first '.' is the source on your machine (your project folder).
# The second '.' is the destination in the container (which is /app).
COPY . .

# 6. Expose the port your application will run on.
# This is documentation for Kubernetes and other tools.
EXPOSE 8008

# 7. Define the command to run your application.
# IMPORTANT: We remove '--reload' as it is for development, not production.
CMD ["uvicorn", "api.ollama_wrapper:app", "--host", "0.0.0.0", "--port", "8008"]
