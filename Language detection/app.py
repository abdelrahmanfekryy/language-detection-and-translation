import os
from fastapi import FastAPI, UploadFile, File, Request,Form,Body
from fastapi.responses import JSONResponse
from config import settings
import uvicorn
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the saved model
with open(f"{dir_path}/Models/vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

with open(f"{dir_path}/Models/classifier.pkl", 'rb') as file:
    model = pickle.load(file)


def predict(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

@app.post("/pred")
async def create_upload_file(request: Request):
    req = await request.json()
    return JSONResponse(predict(req["text"]))

@app.post("/form")
async def create_upload_file(form: str = Form(...)):
    return JSONResponse(predict(form))

@app.post("/input")
def input_request(payload: dict = Body(...)):
    return JSONResponse(predict(payload["text"]))

if __name__ == "__main__":
    uvicorn.run(f"app:app", reload=True, host=settings.DOMAIN, port=settings.PORT)
