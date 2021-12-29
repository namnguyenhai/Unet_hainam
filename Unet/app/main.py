from fastapi import FastAPI,File,Response
from utils import transfrom,predict
import uvicorn

app = FastAPI()

@app.post("/upimg/")
def up_file(file: bytes=File(...)):
    tran = transfrom(file)
    pre = predict(tran)
    return Response(content=pre,media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("main:app",port=8000,reload=True,host="0.0.0.0")