# Step 1: Use a lightweight Python image
FROM python:3.10-slim

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Copy only the essentials (reduces image size)
COPY requirements.txt .

# Step 4: Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your source code
COPY . .

# Step 6: Expose the port your app runs on (usually 8000 for FastAPI)
EXPOSE 8000

# Step 7: Run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
