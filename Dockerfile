FROM python:3.7-slim

WORKDIR /kaggle

#do requirements before code because they change less
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
#NOTE: do NOT use pip-tools within here because the point is to use
#fixed dependencies

COPY . .

