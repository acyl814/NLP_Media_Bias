from fastapi import FastAPI

app = FastAPI(
    title="NLP Media Bias API",
    description="Backend NLP pour l'analyse des biais médiatiques",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Backend NLP Media Bias is running"}
