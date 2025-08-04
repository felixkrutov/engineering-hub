--- a/worker/Dockerfile
+++ b/worker/Dockerfile
@@ -5,6 +5,9 @@
 WORKDIR /app
 
 COPY requirements.txt .
+
+# Install system dependencies needed by python-magic
+RUN apt-get update && apt-get install -y libmagic1
+
 RUN pip install --no-cache-dir -r requirements.txt
 
 COPY . .
