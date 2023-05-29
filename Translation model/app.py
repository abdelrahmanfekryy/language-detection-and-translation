import os
from fastapi import FastAPI, UploadFile, File, Request,Form,Body
from fastapi.responses import JSONResponse
from config import settings
import uvicorn
import dill
import tensorflow as tf
from utils.helper import Tokenizer, Encoder,Decoder, preprocess_sentence


app = FastAPI()
dir_path = os.path.dirname(os.path.realpath(__file__))


with open(f"{dir_path}/Models/AR_Tokenizer.pkl", 'rb') as file:
    AR = dill.load(file)

with open(f"{dir_path}/Models/EN_Tokenizer.pkl", 'rb') as file:
    EN = dill.load(file)

with open(f"{dir_path}/Models/encoder.json", "r") as file:
    encoder_json = file.read()

# Create an instance of your subclassed model
encoder = tf.keras.models.model_from_json(encoder_json, custom_objects={'Encoder': Encoder})

encoder(tf.zeros((64, 1)),tf.zeros((64, 1024)))
# Load the model's weights
encoder.load_weights(f"{dir_path}/Models/encoder_en2ar.h5")


with open(f"{dir_path}/Models/decoder.json", "r") as file:
    decoder_json = file.read()

# Create an instance of your subclassed model
decoder = tf.keras.models.model_from_json(decoder_json, custom_objects={'Decoder': Decoder})

decoder(tf.zeros((64, 1)),tf.zeros((64, 1024)),tf.zeros((64, 1, 1024)))
# Load the model's weights
decoder.load_weights(f"{dir_path}/Models/decoder_en2ar.h5")

def predict(text,lang):
    
    text = preprocess_sentence(text,lang)

    inputs = EN.texts_to_sequences([text])[0]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=EN.maxlen, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, 1024))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([AR.word2idx['<start>']], 0)

    for t in range(AR.maxlen):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.random.categorical(tf.exp(predictions), num_samples=1)[0][0].numpy()


        if AR.idx2word[predicted_id] == '<end>':
            return result
        
        result += AR.idx2word[predicted_id] + ' '


        dec_input = tf.expand_dims([predicted_id], 0)

    return result


@app.post("/pred")
async def create_upload_file(request: Request):
    req = await request.json()
    return JSONResponse(predict(req["text"],"en"))

@app.post("/form")
async def create_upload_file(form: str = Form(...)):
    return JSONResponse(predict(form,"en"))

@app.post("/input")
def input_request(payload: dict = Body(...)):
    return JSONResponse(predict(payload["text"],"en"))


if __name__ == "__main__":
    uvicorn.run(f"app:app", reload=True, host=settings.DOMAIN, port=settings.PORT)
