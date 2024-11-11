FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install PyMuPDF==1.24.9
RUN pip install openai==0.28


COPY . /app/

COPY extract_questions.py /app/

CMD ["python", "app.py"]
