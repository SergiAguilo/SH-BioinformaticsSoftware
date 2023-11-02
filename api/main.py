from fastapi import FastAPI, Query, Path
from pydantic import BaseModel, AnyUrl
from typing import Union
from typing_extensions import Annotated
import json
import os

from api.predict import predict


app = FastAPI(
    title="Bioinformatics Readme Classification API",
    description="This API allows to predict if a given readme file belongs to a bioinformatics project or not.",
    version="0.1.0",
    contact={
        "name": "Eva Martin",
        "email": "eva.martin@bsc.es",
        },
    license_info={
        "name": "MIT",
        "url": "https://spdx.org/licenses/MIT.html",
    }
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


class readme(BaseModel):
    content: str = None
    url: AnyUrl = None

@app.post("/predict")
def predict_bio(readme : readme = None):
    if readme.content is None:
        # return error
        return "No readme content provided", 400
        
    else:
    
        # Load the model
        # 🚧 PLUG THE MODEL HERE
        model = ""
        # Make prediction
        prediction, confidence = predict(readme.content)
        # Return the prediction

        return {
            "prediction": prediction,
            "confidence": confidence,
            "url": readme.url
        }

