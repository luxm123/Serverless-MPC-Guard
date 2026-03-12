FROM alpine:3.18
RUN apk add --no-cache python3 py3-pip
RUN pip install flask --break-system-packages
COPY app.py /app.py
WORKDIR /
EXPOSE 5000
CMD ["python3", "app.py"]
