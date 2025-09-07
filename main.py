import os
import uvicorn
import cv2
import numpy as np
import base64
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración del modelo
MODEL_PATH = "model_rigido.pt"
CONF_THRESHOLD = 0.25 # Mantenemos los thresholds originales
IOU_THRESHOLD = 0.2   # Mantenemos los thresholds originales

# Verificar si el modelo existe al inicio
if not os.path.exists(MODEL_PATH):
    logger.error(f"Modelo no encontrado en {MODEL_PATH}")
    # En un entorno de producción, asegúrate de que el modelo esté en la ruta correcta.
    # Aquí lanzamos un error para que el despliegue falle si el modelo no está.
    raise FileNotFoundError(f"El archivo {MODEL_PATH} no existe. Asegúrate de que esté en la misma carpeta que main.py")

# Cargar el modelo YOLO al iniciar la aplicación
try:
    model = YOLO(MODEL_PATH)
    logger.info("Modelo YOLO cargado exitosamente")
except Exception as e:
    logger.error(f"Error cargando el modelo: {e}")
    raise RuntimeError(f"Error cargando el modelo: {e}")

# Crear la aplicación FastAPI
app = FastAPI(
    title="YOLO Object Detection API",
    description="API para detección de objetos usando YOLOv8 con un frontend web móvil",
    version="1.0.0"
)

# Configurar CORS para permitir solicitudes desde cualquier origen (necesario para el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, considera restringir esto a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar el directorio 'static' para servir el frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

def process_image(img: np.ndarray, conf_threshold: float = CONF_THRESHOLD, iou_threshold: float = IOU_THRESHOLD) -> Dict[str, Any]:
    """
    Procesa una imagen con el modelo YOLOv8, dibuja las cajas delimitadoras
    y extrae las etiquetas únicas.
    """
    try:
        # Realizar inferencia con YOLO
        results = model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,      # No guardar imágenes en el servidor
            show=False,      # No mostrar imágenes en el servidor
            device="cpu",    # Forzar CPU para despliegues en servicios gratuitos
            verbose=False    # Menos logs de YOLO
        )

        detections = []
        unique_labels = set() # Para almacenar las etiquetas de clase únicas
        im_rgb = None

        for r in results:
            # Dibuja las cajas delimitadoras y etiquetas en la imagen
            # r.plot() devuelve una imagen en formato BGR de OpenCV
            im_bgr = r.plot()
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) # Convertir a RGB para consistencia si se desea

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [x1, y1, x2, y2],
                        "class_id": class_id
                    })
                    unique_labels.add(class_name) # Añadir a las etiquetas únicas

        # Convertir la imagen procesada (con cajas) a base64 para enviarla al frontend
        if im_rgb is not None:
            # OpenCV imencode espera BGR para JPEG, así que convertimos RGB de nuevo a BGR
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode("utf-8")
        else:
            # Si no hay detecciones, devolver la imagen original en base64
            _, buffer = cv2.imencode(".jpg", img)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "detections": detections,
            "image_base64": img_base64,
            "num_detections": len(detections),
            "unique_labels": list(unique_labels) # Convertir el conjunto a lista para serialización JSON
        }

    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Sirve el archivo HTML del frontend de la aplicación.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict/upload")
async def predict_from_upload(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.0, le=1.0, description="Umbral de confianza"),
    iou: float = Query(IOU_THRESHOLD, ge=0.0, le=1.0, description="Umbral de IoU")
):
    """
    Endpoint para predecir objetos en una imagen subida por el usuario.
    """
    try:
        logger.info(f"Procesando imagen subida: {file.filename}")

        # Verificar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        # Leer el contenido del archivo de imagen
        contents = await file.read()
        image_bytes = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

        # Procesar la imagen con el modelo YOLO
        result = process_image(img, conf, iou)

        # Devolver los resultados, incluyendo la imagen con cajas, etiquetas únicas y detalles
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "source": "upload",
            "filename": file.filename,
            "file_size": len(contents),
            "parameters": {
                "confidence_threshold": conf,
                "iou_threshold": iou
            },
            **result # Esto incluye 'detections', 'image_base64', 'num_detections', 'unique_labels'
        }

    except Exception as e:
        logger.error(f"Error en predict_from_upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint de salud para verificar que la API está funcionando."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    # Obtener el puerto de la variable de entorno o usar 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)