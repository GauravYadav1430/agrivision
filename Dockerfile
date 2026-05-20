# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the cloud server
WORKDIR /app

# Copy all your files into the cloud server
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the uploads folder just in case
RUN mkdir -p static/uploads

# Expose the port Hugging Face uses
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py"]