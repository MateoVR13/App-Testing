import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Web App para Detección de Patologías de Pavimento Rígido",
    description="Frontend para la detección de objetos, utilizando un servicio de predicción.",
    version="1.0.0"
)

# Configurar CORS (aunque el frontend no hace llamadas directas a este backend, es buena práctica)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permitir desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar el directorio 'static' para servir el frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Sirve el archivo HTML del frontend de la aplicación.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
