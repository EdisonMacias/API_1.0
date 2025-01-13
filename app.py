from flask import Flask, request, jsonify
from ultralytics import YOLO
import io
import base64
from PIL import Image

# Inicializar Flask y cargar el modelo YOLO
app = Flask(__name__)
model = YOLO("C:/Users/ediso/OneDrive/Escritorio/API_1.0/model/best.pt")

# Diccionario de detalles según la clase detectada
class_details = {
    "Adultas": "La etapa adulta es la fase final del ciclo de vida de Diatraea. Aquí, los insectos ya están maduros.",
    "Con_daño": "Esta clase indica que la planta está siendo afectada por Diatraea, lo que podría reducir su rendimiento.",
    "Huevos": "Los huevos de Diatraea son una etapa temprana en su ciclo de vida. La detección temprana es crucial para el control.",
    "Larvas": "Las larvas de Diatraea pueden causar daños significativos al maíz, alimentándose de las plantas.",
    "Maiz_sano": "Este es un maíz saludable, sin signos visibles de daño por Diatraea.",
    "Otras": "Esta clase cubre otras plagas que podrían estar afectando al maíz."
}

def image_to_base64(image):
    """Convierte una imagen a base64"""
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No se subió ningún archivo", 400

    file = request.files['file']
    if file.filename == '':
        return "No se seleccionó ningún archivo", 400

    try:
        # Leer la imagen subida por el usuario
        image = Image.open(file.stream)

        # Realizar detección con YOLO
        results = model(image)

        # Verificar si hay detecciones
        if len(results[0].boxes) == 0:
            # No se detectaron objetos, devolver la imagen original y mensaje
            img_base64 = image_to_base64(image)

            response = {
                "status": "success",
                "image_base64": img_base64,
                "class_names": ["Clase no detectada"],
                "details": ["No se detectó objeto o plaga relacionada al maíz"]
            }

            return jsonify(response), 200

        # Generar la imagen anotada si se detectaron objetos
        annotated_image = results[0].plot()

        # Convertir la imagen anotada a un objeto en memoria
        annotated_image_pil = Image.fromarray(annotated_image)

        # Convertir la imagen anotada a base64
        annotated_image_base64 = image_to_base64(annotated_image_pil)

        # Obtener los detalles de las clases detectadas
        class_names = []
        details = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            class_names.append(class_name)

            # Obtener el detalle de la clase
            detail = class_details.get(class_name, "Sin información adicional.")
            details.append(detail)

        # Respuesta con la imagen en base64 y detalles de las clases detectadas
        response = {
            "status": "success",
            "image_base64": annotated_image_base64,
            "class_names": class_names,
            "details": details
        }

        return jsonify(response)

    except Exception as e:
        # Detallar el error para una mejor depuración
        print(f"Error: {str(e)}")
        return f"Ocurrió un error durante el procesamiento: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=False)
