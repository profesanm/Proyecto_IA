# LUMINA: Evaluador de Apropiación de Retroalimentación Efectiva

Herramienta predictiva basada en Inteligencia Artificial que clasifica a docentes en nivel Bajo, Medio o Alto de apropiación de retroalimentación efectiva.

**Modelo final**: LogisticRegression optimizado (F1-macro ≈ 0.90)  
**Datos**: 250 docentes evaluados  
**Variables clave**:
nivel_formacion_retroalimentacion_normalized,
concepto_retroalimentacion_count_normalized,
relevancia_retroalimentacion_normalized, 
momento_entrega_retroalimentacion_count_normalized,
tecnicas_retroalimentacion_count_normalized, 
tiempo_retroalimentacion_clase_normalized,
frecuencia_tecnologias_retroalimentacion_normalized,
herramientas_tecnologicas_count_normalized,

## Cómo ejecutarlo (2 minutos)
Abre una terminal en esta carpeta y ejecuta:

```bash   
  pip install -r requirements.txt
  streamlit run app.py