from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from datetime import datetime

class Item(BaseModel):
    text: str

app = FastAPI()
sentiment_pipeline = pipeline("sentiment-analysis")

@app.get("/")
def greet_json():
    try:
        formatted_time_string =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_time_string = "Current Date Time is: ".join(formatted_time_string)
        return {"Hello": formatted_time_string}
    except Exception as e:
        return {"Hello": f"Error occured: {e}"}

@app.post("/sentiment")
def get_sentiment(item: Item):
    return sentiment_pipeline(item.text)[0]

@app.get("/health")
def health_check():
    return {"status": "ok"} 

@app.get("/healthz")
def health_check_z():
    try:
        formatted_time_string =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_time_string = "Ok. Current Date Time is: ".join(formatted_time_string)
        return {"Status": formatted_time_string}
    except Exception as e:
        return {"Status": f"Error occured: {e}"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8080)
