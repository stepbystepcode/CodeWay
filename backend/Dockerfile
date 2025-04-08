FROM agnohq/python:3.12

WORKDIR /app

COPY pyproject.toml .
COPY gotrue-2.12.0 ./gotrue-2.12.0
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
RUN pip install uv 
RUN uv venv
RUN uv pip install -e .

COPY . .

CMD ["granian", "--interface", "asgi", "main:app", "--host", "0.0.0.0", "--port", "8000"]