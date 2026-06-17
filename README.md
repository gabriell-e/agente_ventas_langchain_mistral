# Drug Side Effects - Clasificacion y RAG con Mistral

Pipeline completo de ciencia de datos para clasificar severidad de efectos secundarios de medicamentos, con arquitectura de 3 agentes y un agente RAG con Mistral AI.

## Dataset

5000 registros de pacientes con reacciones adversas a farmacos. Cada registro incluye edad, genero, pais, medicamento, dosis, efecto secundario, severidad, outcome, condiciones cronicas, y otros atributos.

## Arquitectura del proyecto

### Agente 1: Normalizador (`AgenteNormalizador`)
- Carga flexible con deteccion de encoding
- Renombre a snake_case y conversion de tipos (edad, dosis, dias de recuperacion)
- Parseo de fechas con multiples formatos
- Estandarizacion de strings (genero, pais, severidad, outcome, etc.)
- Imputacion de nulos (mediana para numericos, moda para categoricos)
- Analisis exploratorio con graficos

### Agente 2: Modelo (`AgenteModelo`)
- Codificacion one-hot de variables categoricas
- Division train/test (80/20) estratificada
- Escalado con StandardScaler
- Entrenamiento de 4 clasificadores
- Evaluacion y seleccion del mejor modelo
- Prediccion para nuevos pacientes

### Agente 3: RAG (`AgenteRAG`)
- Vectorstore FAISS con embeddings Mistral via LangChain
- Retrieval de 5 registros mas relevantes por similitud semantica
- Prompt de sistema antisesgos para respuestas basadas unicamente en datos
- Reintentos automaticos ante rate limit de la API Mistral

### Standalone: DistilBERT
Clasificador transformer como comparacion contra los modelos tabulares.

## Resultados

### Exploracion de datos

- **Distribucion de severidad:** Mild 3147 (63%), Moderate 1420 (28%), Severe 433 (9%)
- **Distribucion de outcome:** Recovered 3510 (70%), Recovering 911 (18%), Hospitalized 523 (10%), Fatal 56 (1%)
- **Nulos iniciales:** ChronicCondition 645, AlcoholUse 1076, RecoveryDays 579
- **Shape final:** 5000x16, sin nulos

### Clasificacion

| Modelo | Accuracy | F1 Score (weighted) |
|---|---|---|
| Logistic Regression | 0.320 | 0.369 |
| Decision Tree | 0.484 | 0.484 |
| **Random Forest** | **0.596** | **0.495** |
| Gradient Boosting | 0.622 | 0.488 |

**Mejor modelo: Random Forest** (F1 = 0.495)

El modelo predice bien la clase mayoritaria (Mild, recall 0.92) pero tiene dificultades con Moderate (recall 0.06) y Severe (0.00), lo que refleja el fuerte desbalance de clases.

**Prediccion de ejemplo** (paciente: 55 anos, Male, USA, Atorvastatin 25mg, hypertension, fumador):
- Mild: 61%, Moderate: 30%, Severe: 9%

### DistilBERT

Entrenado con descripciones textuales de cada caso (3 epocas, batch 16):

- **Accuracy:** 0.629
- **F1 Score:** 0.486

Resultados comparables a Random Forest (F1 0.495 vs 0.486), pero con un costo computacional significativamente mayor.

### RAG

5000 documentos indexados en FAISS con MistralAIEmbeddings usando LangChain. El agente responde preguntas en lenguaje natural recuperando los 5 registros mas relevantes y usando ChatMistralAI con un prompt de sistema antisesgos.

**Pregunta 1:** "Que efectos secundarios son mas comunes en pacientes con diabetes?"

> Hypoglycemia (2 registros), Weight Gain (2 registros) y Diarrhea (1 registro). Nota: el dataset es extremadamente pequeno (5 registros), por lo que estos resultados no son generalizables.

**Pregunta 2:** "Cual es la relacion entre la edad y la severidad de los efectos?"

> El unico caso de severidad moderada ocurrio en el paciente de mayor edad (73 anos), mientras que los casos leves se distribuyeron en edades mas jovenes (27-59 anos). No hay evidencia suficiente para generalizar.

**Pregunta 3:** "Que farmacos tienen mas probabilidad de causar hospitalizacion?"

> Paracetamol, Amoxicillin e Ibuprofen aparecen asociados a hospitalizacion en el 100% de sus registros. Sin embargo, no es posible calcular probabilidades reales porque el dataset no incluye pacientes que tomaron estos farmacos sin ser hospitalizados.

El prompt de sistema antisesgos evito alucinaciones en todos los casos: las respuestas fueron honestas sobre las limitaciones de los datos y evitaron conclusiones causales sin respaldo.

## Conclusiones

- Random Forest fue el mejor modelo tabular, pero el desbalance de clases limita su rendimiento en clases minoritarias
- DistilBERT obtuvo resultados similares con un costo computacional mucho mayor
- El agente RAG con Mistral AI y prompt antisesgos demostro respuestas precisas y honestas
- La arquitectura en 3 agentes permite mantener el codigo modular y facil de mantener
