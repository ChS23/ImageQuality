from PIL import Image
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse
from .metrics import calculate_metric, metrics

app = FastAPI()


@app.get("/")
async def home():
    return RedirectResponse(url="/docs")


@app.post("/loss")
async def predict(
        package: str = Form(...),
        metric: str = Form(...),
        image1: UploadFile = UploadFile(...),
        image2: UploadFile = UploadFile(...),
):
    image1 = Image.open(image1.file).convert('RGB')
    image2 = Image.open(image2.file).convert('RGB')
    return {"value": calculate_metric(image1, image2, package, metric)}


@app.post("/metrics")
async def get_metrics(package: str = Form(...)):
    metrics_list = metrics[package].keys()

    if not metrics_list:
        return HTTPException(status_code=400, detail="Package not found")

    return {"metrics": list(metrics_list)}


@app.post("/packages")
async def get_packages():
    return {"packages": list(metrics.keys())}
