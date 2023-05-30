import os
from fastapi import FastAPI, UploadFile, File, Request,Form,Body
from fastapi.responses import JSONResponse
from config import settings
import uvicorn
import dill
import tensorflow as tf
from utils.helper import Tokenizer, Encoder,Decoder, preprocess_sentence
from langdetect import detect_langs
import numpy as np


app = FastAPI()
dir_path = os.path.dirname(os.path.realpath(__file__))


with open(f"{dir_path}/Models/AR_Tokenizer.pkl", 'rb') as file:
    AR = dill.load(file)

with open(f"{dir_path}/Models/EN_Tokenizer.pkl", 'rb') as file:
    EN = dill.load(file)



###########

with open(f"{dir_path}/Models/encoder_en2ar.json", "r") as file:
    encoder_json = file.read()

# Create an instance of your subclassed model
encoder_en2ar = tf.keras.models.model_from_json(encoder_json, custom_objects={'Encoder': Encoder})

encoder_en2ar(tf.zeros((64, 1)),tf.zeros((64, 1024)))
# Load the model's weights
encoder_en2ar.load_weights(f"{dir_path}/Models/encoder_en2ar.h5")


with open(f"{dir_path}/Models/decoder_en2ar.json", "r") as file:
    decoder_json = file.read()

# Create an instance of your subclassed model
decoder_en2ar = tf.keras.models.model_from_json(decoder_json, custom_objects={'Decoder': Decoder})

decoder_en2ar(tf.zeros((64, 1)),tf.zeros((64, 1024)),tf.zeros((64, 1, 1024)))
# Load the model's weights
decoder_en2ar.load_weights(f"{dir_path}/Models/decoder_en2ar.h5")

####################

with open(f"{dir_path}/Models/encoder_ar2en.json", "r") as file:
    encoder_json = file.read()

# Create an instance of your subclassed model
encoder_ar2en = tf.keras.models.model_from_json(encoder_json, custom_objects={'Encoder': Encoder})

encoder_ar2en(tf.zeros((64, 1)),tf.zeros((64, 1024)))
# Load the model's weights
encoder_ar2en.load_weights(f"{dir_path}/Models/encoder_ar2en.h5")


with open(f"{dir_path}/Models/decoder_ar2en.json", "r") as file:
    decoder_json = file.read()

# Create an instance of your subclassed model
decoder_ar2en = tf.keras.models.model_from_json(decoder_json, custom_objects={'Decoder': Decoder})

decoder_ar2en(tf.zeros((64, 1)),tf.zeros((64, 1024)),tf.zeros((64, 1, 1024)))
# Load the model's weights
decoder_ar2en.load_weights(f"{dir_path}/Models/decoder_ar2en.h5")

#######################

def langWithThresh(text,thresh):
    langs = np.array([[lang.lang, lang.prob] for lang in detect_langs(text)])
    idx = langs[:,1].argmax()
    if float(langs[:,1][idx]) > thresh:
        return langs[:,0][idx]

def predict(text):
    lang = langWithThresh(text,0.5)
    if lang == "en" or lang == "ar":
        encoder = {"ar":encoder_ar2en,"en":encoder_en2ar}[lang]
        decoder = {"ar":decoder_ar2en,"en":decoder_en2ar}[lang]
        input_tok = {"ar":AR,"en":EN}[lang]
        output_tok = {"ar":EN,"en":AR}[lang]
        text = preprocess_sentence(text,lang)
    else:
        return "Error: Unknown/Unsupported Language"
    
    inputs = input_tok.texts_to_sequences([text])[0]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=input_tok.maxlen, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, 1024))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([output_tok.word2idx['<start>']], 0)

    for t in range(output_tok.maxlen):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.random.categorical(tf.exp(predictions), num_samples=1)[0][0].numpy()

        if output_tok.idx2word[predicted_id] == '<end>':
            
            return result
        
        result += output_tok.idx2word[predicted_id] + ' '


        dec_input = tf.expand_dims([predicted_id], 0)

    return result


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
