import os
from fastapi        import FastAPI, Request
from pydantic       import BaseModel
from transformers   import pipeline
from datetime       import datetime
from pathlib        import Path
from dotenv         import load_dotenv
load_dotenv()
from contextlib     import asynccontextmanager


@asynccontextmanager
asnyc def lifespan(app: FastAPI):
    print("Application startup: Initializing resources...")
    
    model_path = os.getenv("MODEL_PATH")
    
    # Use a default model if MODEL_PATH is not set or doesn't exist
    if model_path and os.path.exists(model_path):
        print(f"Loading model from local path: {model_path}")
        app.state.sentiment_pipeline = pipeline("sentiment-analysis", model=model_path)
    else:
        default_model = "distilbert-base-uncased-finetuned-sst-2-english"
        print(f"Loading default model from Hugging Face Hub: {default_model}")
        app.state.sentiment_pipeline = pipeline("sentiment-analysis", model=default_model)
    
    yield
    
    # Code executed on shutdown
    print("Application shutdown: Cleaning up resources...")
    app.state.sentiment_pipeline = None


app = FastAPI(lifespan=lifespan)

class Item(BaseModel):
    text: str


index = 0

@app.get("/")
def greet_json():
    global index
    try:
        index += 1
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"greeting": "hello", "timestamp": f"current time is : {time_now}", "calls": index}
    except Exception as e:
        return {"Hello": f"Error occured: {e}"}

@app.post("/sentiment")
def get_sentiment(item: Item, request: Request):
    # Access the pipeline from the application state
    pipeline = request.app.state.sentiment_pipeline
    return pipeline(item.text)[0]

# New add endpoint - simple version
@app.get("/add")
def add_numbers(num1: float, num2: float):
    result = num1 + num2
    return {"num1": num1, "num2": num2, "result": result}


@app.get("/health")
def health_check():
    return {"status": "ok"} 

@app.get("/healthz")
def health_check_z():
    try:
        time_now =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_now = "Ok. Current Date Time is: " + time_now
        return {"Status": time_now}
    except Exception as e:
        return {"Status": f"Error occured: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)