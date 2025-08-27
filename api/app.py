
# ----- Kütüphane importları -----
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from aimped.nlp.tokenizer import sentence_tokenizer, word_tokenizer
from aimped.nlp.pipeline import Pipeline
import warnings
import json, os, requests, logging, sys
from fastapi.middleware.cors import CORSMiddleware



# ----- Uyarıları kapat -----
warnings.filterwarnings('ignore')


print("Model dependencies imported")


# ----- FastAPI uygulaması başlat -----
app = FastAPI(
    title="Ar86Bat/multilang-pii-ner",
    version="0.1.0",
    description="A FastAPI application for pii identification with Named Entity Recognition (NER)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gerekirse sadece 'http://localhost:3000'
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----- API Request/Response Modelleri -----

class PredictionRequest(BaseModel):
    text: List[str]
    

class PredictionResponse(BaseModel):
    status: bool 
    data_type: str
    output: List[Dict] 

class HealthResponse(BaseModel):
    name: str = Field(..., description="Model name.")
    status: str = Field(default="OK", description="Health status of the model.")


class KFServeHealthDeidNerModel:
    def __init__(self):
        self.device = "cpu"
        self.ready = False
        self.data_type = "data_json"
        self.model = None
        self.tokenizer = None
        self.pipe = None

        self.load()


    def load(self):
        """
        Model ve Tokenizer yüklenir, Pipeline hazırlanır.
        """
        try:
            self.model = AutoModelForTokenClassification.from_pretrained("Ar86Bat/multilang-pii-ner")
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("Ar86Bat/multilang-pii-ner")
            self.pipe = Pipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)
            self.ready = True
            print(f"Model loaded successfully: Ar86Bat/multilang-pii-ner")



        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    def get_prediction(self, text: str):
        """
        Girilen bir cümlede NER sonucu ve masking/faking uygular.
        """
        sentences = sentence_tokenizer(text, "german")
        sents_tokens_list = [sent.split(" ") for sent in sentences]
        tokens, preds, probs, begins, ends = self.pipe.ner_result(text=text, sents_tokens_list=sents_tokens_list, sentences=sentences)

        white_label_list = labels = [
        "AGE",
        "BUILDINGNUM",
        "CITY",
        "CREDITCARDNUMBER",
        "DATE",
        "DRIVERLICENSENUM",
        "EMAIL",
        "GENDER",
        "GIVENNAME",
        "IDCARDNUM",
        "PASSPORTNUM",
        "SEX",
        "SOCIALNUM",
        "STREET",
        "SURNAME",
        "TAXNUM",
        "TELEPHONENUM",
        "TIME",
        "TITLE",
        "ZIPCODE"]

        results = self.pipe.chunker_result(text, white_label_list, tokens, preds, probs, begins, ends)
        return results
        

    def predict(self, request: PredictionRequest):
        """
        Birden fazla input text'i işler.
        """
        if not self.ready:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        if not request.text:
            print.error("No input text provided")
            raise HTTPException(status_code=400, detail="No input text provided")
        print(request.text)
        outputs = []
        for text in request.text:
            preds = self.get_prediction(text)
            outputs.extend(preds)
        print("Prediction completed successfully")
        return {"status": True, "data_type": self.data_type, "output": outputs}

# ----- Model Instance -----
model_instance = KFServeHealthDeidNerModel()

# ----- API Endpointleri -----

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"name": "model_instance", "status": "Ready" if model_instance.ready else "Loading"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        return model_instance.predict(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))