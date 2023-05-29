## Language Translation

### Prerequisites

- Python v3.8

## Install dependencies

```shell
> pip install --upgrade virtualenv
> virtualenv <env-name> --python=python3.8
> "./<env-name>/Scripts/activate"
> pip install -r requirements.txt
```

## Run instructions 

```shell
> python app.py
```

## invoke endpoint
```python
import requests
import json
r = requests.post("http://127.0.0.1:8080/pred/",data=json.dumps({"text":<the-text-to-be-detected>,"lang":<lang-of-text>}))
r.text
```
or

FastAPI documentation page: http://localhost:8080/docs

## Run instructions using docker

```shell
> docker pull abdelrahmanfekryy/language-translation:1.0.0
> docker run -p8080:8080 abdelrahmanfekryy/language-translation
```