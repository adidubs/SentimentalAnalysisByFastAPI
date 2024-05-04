from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sentimental as a
app = FastAPI(title="Text Classification API")



class Item(BaseModel):
    comment: str
    

#http://127.0.0.1:8000/
#/predict

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(item: Item):
    try:
        # Vectorize the incoming text
        vect_text = a.vectorizer.transform([item.comment])
        # Predict using all three models
        pred_nb = a.nb_classifier.predict(vect_text)
        pred_svm = a.svm_classifier.predict(vect_text)
        pred_rf = a.rf_classifier.predict(vect_text)
        
        return {
            "Naive_Bayes": str(pred_nb[0]),
            "SVM": str(pred_svm[0]),
            "Random_Forest": str(pred_rf[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
@app.get("/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    # Stub: Replace with real metrics retrieval logic
    if model_name.lower() == "svm":
        metrics = {
            "accuracy": 0.3923,
            "classification_report": "Precision: 0.40, Recall: 0.39, F1-score: 0.39"
        }
    elif model_name.lower() == "random_forest":
        metrics = {
            "accuracy": 0.4478,
            "classification_report": "Precision: 0.40, Recall: 0.45, F1-score: 0.39"
        }
    elif model_name.lower() == "naive_bayes":
        metrics = {
            "accuracy": 0.4542,
            "classification_report": "Precision: 0.43, Recall: 0.45, F1-score: 0.43"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Metrics for model {model_name} not found.")
    return metrics

#uvicorn your_script_name:app --reload  # `your_script_name` should be replaced with the name of your Python file.

#uvicorn fastAPISetUp:app --reload
