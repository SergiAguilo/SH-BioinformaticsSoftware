FROM python:3.9
COPY ./api api

WORKDIR api

RUN python -m pip install -r requirements.txt --no-dependencies

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5300"]

EXPOSE 5300
