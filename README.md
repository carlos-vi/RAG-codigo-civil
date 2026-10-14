# RAG sobre el Código Civil español

Asistente de consulta del **Código Civil** (texto consolidado del BOE) basado en RAG: busca los artículos relevantes para una pregunta y un LLM responde **solo con lo que dicen esos artículos**, citándolos. El LLM y los embeddings se sirven desde la API gratuita de [build.nvidia.com](https://build.nvidia.com); el reranker se ejecuta en local.

> **Aviso:** es una herramienta de consulta del texto legal, no asesoramiento jurídico. Solo cubre el Código Civil (sin leyes complementarias ni jurisprudencia) y puede equivocarse.

## 1. Cómo funciona

```
                 pregunta
                    |
        +-----------+------------+
        v                        v
  pregunta original   reescrita en términos jurídicos (LLM)
        |                        |
        v                        v
   búsqueda híbrida         búsqueda híbrida
  (BM25 + embeddings)      (BM25 + embeddings)
        |                        |
        +-----------+------------+
                    v
     10 primeros artículos de cada una
                    |
                    v
   reranker local (cross-encoder, mejor
   puntuación con cualquiera de las dos)
                    |
                    v
   + artículos citados por número, primero
                    |
                    v
            top 6 artículos
                    |
                    v
          LLM (solo con ese contexto)
                    |
                    v
             respuesta citada
```

1. **Ingesta** (`src/ingest.py`): descarga el Código Civil del BOE (API de datos abiertos, XML) y lo trocea **por estructura legal**. Cada artículo vigente es un *padre* con su jerarquía (libro › título › capítulo › sección); se usa la última versión de cada artículo y se descartan los derogados. Cada padre se divide en *hijos* de ~700 caracteres, que son lo que se indexa.
2. **Artículos citados por número** (`cited_articles` en `src/retrieval.py`): si la pregunta cita un artículo ("artículo 1", "art. 1902", "art 94 bis", "artículos 1, 2 y 3"), se recupera directamente y va el primero.
3. **Reescritura de la consulta** (`rewrite_query`): un LLM traduce la pregunta a la terminología del Código ("¿cuánto tiempo tengo para reclamar una deuda?" → "plazo de prescripción de la acción personal para reclamar una deuda"), sin números de artículo para que no pueda inventarlos. Las preguntas que no tratan de derecho civil no se reescriben. Si la reescritura falla o tarda más de 60 s, se sigue solo con la pregunta original. Se desactiva con `REWRITE_QUERY = False` en `src/config.py`.
4. **Recuperación híbrida**: BM25 (búsqueda léxica) y búsqueda densa por embeddings, 20 fragmentos por rama, fusionados con RRF ponderado (0,4 / 0,6), con la pregunta original y con la reescrita. Los fragmentos se expanden a su artículo completo y pasan al reranker los 10 primeros artículos de cada consulta. BM25 tokeniza en minúsculas, sin tildes (conserva la ñ) y sin signos de puntuación, para que `808` en la pregunta coincida con `Artículo 808.` en el texto.
5. **Reranking local** (`src/reranker.py`): un cross-encoder multilingüe lee pregunta y artículo juntos; cada artículo se queda con su mejor puntuación frente a la pregunta original o la reescrita.
6. **Generación** (`src/rag_chain.py`): el LLM recibe los 6 mejores artículos y la pregunta original, y responde únicamente con esos artículos; si la respuesta no está, lo dice.

## 2. Puesta en marcha

Requisitos: Python 3.11 y una clave gratuita de NVIDIA ([build.nvidia.com](https://build.nvidia.com)). El reranker carga `torch` en memoria: con 8 GB de RAM funciona, pero va justo si hay otros procesos pesados abiertos. Los comandos son de Windows; en Linux/macOS, `source .venv/bin/activate` y `cp` en lugar de `copy`.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-rerank.txt
copy .env.example .env                                   # y pega tu NVIDIA_API_KEY en .env
python src/check_models.py                               # comprueba clave, modelos, reranker e índice
python src/ingest.py                                     # descarga, trocea e indexa (~2.250 fragmentos, ~4 min, reanudable)
python src/rag_chain.py --question "¿Qué parte de la herencia constituye la legítima?"
```

`check_models.py` hace una llamada real a cada modelo y explica qué falla y cómo arreglarlo. Es lo primero que conviene ejecutar (ver la sección 5 de por qué).

La ingesta es reanudable porque salta los fragmentos que Chroma ya tiene, contándolos. Por eso, **si cambia el parser o se descarga un XML nuevo, hay que borrar `chroma_db/codigo_civil/` antes de relanzarla**: si no, `children.json` se regenera pero los embeddings antiguos se quedan y BM25 y la búsqueda densa trabajan sobre textos distintos sin avisar. El XML se descarga una sola vez y se reutiliza; borra `data/codigo_civil/` para bajar la versión vigente.

Evaluación y tests (los resultados de la sección 4 se generan con estos comandos):

```
python src/evaluate.py                      # RAGAS con 17 preguntas (~30 min; --limit N para una parte)
python src/eval_retrieval.py                # recuperación con preguntas cotidianas (--split dev|test|cita)
python src/eval_fuera_de_alcance.py         # preguntas que el Código no contesta (revisión manual)
python src/bench_latency.py                 # latencia por etapas
python -m unittest discover -s tests -v     # tests del parser, del tokenizador y de las citas (sin red ni clave)
```

El reranker es opcional: sin `requirements-rerank.txt` el proyecto avisa y funciona con el orden de la fusión híbrida. Se desactiva con `RERANK_MODEL = None` en `src/config.py`.

## 3. Modelos y configuración

Todo está en [src/config.py](src/config.py):

| Papel | Modelo | Dónde se ejecuta |
|---|---|---|
| Generación | `nvidia/nemotron-3-super-120b-a12b` | API de NVIDIA |
| Reescritura de la consulta | el mismo que genera | API de NVIDIA |
| Juez de RAGAS | el mismo que genera | API de NVIDIA |
| Embeddings | `nvidia/nemotron-3-embed-1b` (2048 dim.) | API de NVIDIA |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (117 M, multilingüe) | Local (CPU, ~470 MB, se descarga la primera vez) |

La clave se lee de la variable de entorno `NVIDIA_API_KEY` o de `.env` (ignorado por git). El tier gratuito permite ~40 peticiones por minuto; la ingesta y las evaluaciones van espaciadas para respetarlo.

## 4. Resultados

El índice se construyó con el XML del BOE descargado el 8 de octubre de 2026. La reescritura, la generación y el juez de RAGAS son llamadas a un LLM por API que varían algo entre ejecuciones aunque `temperature`=0, así que al repetir las evaluaciones las cifras cambian ligeramente.

### Evaluación con RAGAS (17 preguntas)

Preguntas formuladas con el vocabulario del Código ([eval/qa_codigo_civil.jsonl](eval/qa_codigo_civil.jsonl)), cada una con una respuesta de referencia. Detalle por pregunta (contextos, respuesta y puntuaciones) en [eval/results_codigo_civil.csv](eval/results_codigo_civil.csv).

| Métrica | Media |
|---|---|
| `context_precision` | **0,93** |
| `context_recall` | **0,88** |
| `faithfulness` | **0,82** (16 preguntas) |
| `answer_relevancy` | **0,62** (16 preguntas) |

En una pregunta el juez no terminó y dio `NaN` en `faithfulness`, y en otra en `answer_relevancy`. El artículo correcto llega al contexto en las 17 preguntas: en 1.ª posición en 15. Las más débiles:

| Pregunta | Qué se observó |
|---|---|
| Pactar libremente un contrato (art. 1255) | `context_precision` 0,33 y `context_recall` 0,5: el artículo entra en 3.ª posición, tras los arts. 1257 y 1258 |
| Legítima (art. 808) | `context_precision` 0,50: el 808 queda 2.º, tras el 809 |
| Interpretación de las normas (art. 3) | `context_recall` 0,0 aunque el art. 3 queda 1.º y contiene literalmente la respuesta de referencia: fallo del juez |
| Ignorancia de las leyes (art. 6) | `context_recall` 0,5 aunque el art. 6 queda 1.º |

**Cómo leerlo, con cautela:**
- **El juez es el mismo modelo que genera**, así que las métricas pueden salir optimistas (un juez de otra familia, GLM 5.3, hacía la evaluación inviable por lenta).
- **El juez es ruidoso:** ejecutar dos veces la evaluación con el mismo sistema da puntuaciones distintas por pregunta (hasta ±0,5 en `faithfulness`). Diferencias de ±0,1 en las medias no son significativas.
- **Estas preguntas se usaron durante el desarrollo** para ajustar `top_k` y el parser, así que las cifras pueden ser optimistas para preguntas nuevas.
- **`answer_relevancy` es la más baja** y no se ha investigado por qué.

### Recuperación con preguntas en lenguaje cotidiano

La gente no pregunta con el vocabulario del Código ("prescripción de las acciones personales") sino como "¿cuánto tiempo tengo para reclamar una deuda?". [eval/qa_cotidiano.jsonl](eval/qa_cotidiano.jsonl) tiene 25 preguntas así, cada una con el artículo o artículos que la responden (comprobados en el texto del corpus), y 5 que citan artículos por número. Se mide sin juez: si alguno de los artículos esperados llega a los 6 que recibe el LLM y en qué posición (MRR: media de 1/posición, 0 si no llega). Detalle en [eval/retrieval_cotidiano.csv](eval/retrieval_cotidiano.csv).

Las 25 preguntas cotidianas están repartidas de forma fija (campo `split`) en *dev*, usadas para decidir el diseño de la reescritura, y *test*, que no se usaron para ajustar nada. Test es la cifra que mejor estima el comportamiento con preguntas nuevas.

| Conjunto | Artículo correcto en el top 6 | En 1.ª posición | MRR |
|---|---|---|---|
| test (12) | **9/12** | 4/12 | 0,46 |
| dev (13) | 11/13 | 9/13 | 0,76 |
| citas por número (5) | 5/5 | 5/5 | 1,00 |

Fallan dejar a un hijo sin herencia (desheredación, arts. 848/853), quién se queda con el perro tras el divorcio (94 bis), la cuota del viudo (834), casarse a los 16 años (46) y los intereses de un préstamo entre amigos (1755). Algunos aciertos quedan en posiciones bajas, como el atropello (1968, 6.º) o el coche con una avería oculta (1484, 4.º).

### Preguntas fuera del Código Civil

9 preguntas que el Código no contesta ([eval/fuera_de_alcance.json](eval/fuera_de_alcance.json)): IVA, multas de tráfico, pena por robo, indemnización por despido, la capital de Francia, una receta de paella, un artículo inexistente (el 5000), uno derogado (el 315) y la fianza de un alquiler (regulada por otra ley). **En 8 de las 9 el modelo declinó responder** y dijo que no estaba en los artículos recibidos, sin inventar contenido. Matices:

- La de la fianza se responde con el art. 1964: la acción para reclamarla prescribe a los 5 años (acción personal sin plazo especial). Es una respuesta fundada en el Código, pero incompleta: no menciona que la fianza del alquiler la regula la Ley de Arrendamientos Urbanos, que el sistema no conoce.
- En la de despido cita el art. 1584 (sobre criados domésticos), con la salvedad de que no establece una indemnización por año trabajado.
- En el art. 315 (derogado) dice que no lo encuentra, sin explicar que fue derogado.
- Es una muestra pequeña, sin métrica automática (se revisa a mano); no se probaron intentos de forzar al modelo (insistir, inyección de instrucciones).

### Latencia

Tiempo por etapa en las 17 preguntas de RAGAS ([eval/latency.csv](eval/latency.csv)), en un portátil de 8 GB sin GPU y con el tier gratuito de NVIDIA:

| Etapa | Mediana | Mín.–máx. |
|---|---|---|
| Reescritura de la consulta (LLM) | 9,7 s | 3,6–20,7 s |
| Búsqueda híbrida (BM25 + embeddings) | 1,4 s | 0,8–3,1 s |
| Reranker local (mediana de 15 candidatos) | 6,7 s | 1,4–21,4 s |
| Generación (LLM) | 4,3 s | 2,8–12,8 s |
| **Total** | **25,1 s** | 12,9–48,3 s |

Las dos llamadas al LLM y el reranker en CPU se llevan casi todo el tiempo. La reescritura incluye las esperas por el límite de peticiones del tier gratuito (error 429, se reintenta hasta tres veces); en una de las 17 preguntas agotó los intentos y se buscó solo con la pregunta original. Los tiempos dependen del equipo y de la carga de la API, y varían bastante entre ejecuciones.

## 5. Decisiones de diseño

- **Validación real de los modelos.** Los endpoints gratuitos de NVIDIA se retiran sin avisar (error `410 Gone`), y el catálogo integrado en la librería `langchain-nvidia-ai-endpoints` sigue listando modelos ya retirados. Solo vale una llamada real: por eso existe `src/check_models.py`.
- **Reranker local con `torch` antiguo.** Al no haber reranker alojado gratuito, se ejecuta en local. En Windows 11 con *Smart App Control* activo, las extensiones nativas de versiones recientes sin reputación quedan bloqueadas (`torch` 2.14, `scikit-learn`, `pandas` 3.x...), por lo que `requirements-rerank.txt` fija versiones antiguas: `torch==2.2.2`, `numpy<2`, `transformers==4.38.2`, `tokenizers==0.15.2`. Se usa `transformers` directamente y no `sentence-transformers`, que arrastra `scikit-learn`. Por la misma razón se fijan `rpds-py==2026.6.3` y `pyarrow<25`. Estas fijaciones no hacen falta en equipos sin Smart App Control, pero no molestan.
- **Particularidades del XML del BOE** que resuelve el parser (y cubren los tests): los rótulos de sección llegan como "preceptos" sin número; hay artículos con sufijo (`94 bis`, `172 ter`); las disposiciones adicionales y transitorias se numeran "Primera", "Segunda"... y se etiquetan con su grupo; cada artículo y cada encabezado conservan todas sus versiones históricas y solo cuenta la vigente.
- **Tokenizador propio para BM25.** El de LangChain por defecto (`text.split()`) deja la puntuación pegada (`808.`, `legítima?`) y distingue mayúsculas, así que no hacía coincidir la pregunta con el texto.
- **Consulta directa por número de artículo.** Buscar un artículo por su número es una consulta exacta, no semántica: el número `1` aparece en cientos de fragmentos y apenas pesa en BM25, y el reranker puntúa parecido de texto, no sabe qué artículo es "el 1". Por eso los artículos citados se recuperan directamente. Si el número no existe o el artículo está derogado, no se añade nada.
- **Reescritura de la consulta.** Se busca con la pregunta original y con la reescrita, y el reranker se queda con la mejor puntuación de las dos: la reescrita rescata preguntas sin vocabulario jurídico, y la original protege los casos en que la reescritura se desvía. Pasar al reranker solo los 10 primeros artículos de cada búsqueda mantiene su coste parecido al de una sola búsqueda. El prompt no incluye ejemplos de instituciones jurídicas, porque el LLM tiende a reciclarlos en preguntas que no tienen que ver.
- **Modelos de razonamiento.** El generador razona antes de responder: con un límite de tokens bajo gasta todo pensando y `content` queda vacío, por eso no se limitan los tokens de salida.
- **`top_k` = 6.** En la evaluación con RAGAS el artículo correcto queda entre los 2 primeros en 16 de las 17 preguntas; 6 da margen para preguntas que necesitan varios artículos.

## 6. Limitaciones

- Solo el Código Civil; sin legislación complementaria, jurisprudencia ni historial de conversación. En temas regulados por otra ley (la fianza del alquiler, por ejemplo) puede responder con el régimen general del Código sin avisar de que existe una ley específica. Interfaz solo por línea de comandos.
- Las evaluaciones son pequeñas: 17 preguntas con RAGAS, 25 cotidianas y 5 de citas, y 9 fuera de alcance. Las cotidianas miden solo la recuperación, no la respuesta final. El conjunto *test* se midió una vez con una versión anterior del prompt de reescritura antes de las cifras finales, así que no es del todo virgen.
- El juez de RAGAS es el mismo modelo que genera y varía entre ejecuciones (ver sección 4).
- La reescritura añade una llamada más al LLM por pregunta y depende de que la API responda (si falla, se busca solo con la pregunta original).
- La consulta directa por número asume que el artículo citado es del Código Civil: en "artículo 14 de la Constitución" se añadiría el art. 14 del Código.
- El reranker trunca pregunta y artículo a 384 tokens (el máximo del modelo es 512): los artículos más largos no se puntúan enteros.
- El contenido de las consultas sale a la API de NVIDIA (el corpus es público; las preguntas, no).
- Las versiones fijadas (`numpy<2`, `torch` 2.2.2...) son frágiles ante actualizaciones.

## 7. Estructura

```
src/config.py                  modelos, rutas y parámetros
src/ingest.py                  BOE → artículos → fragmentos → Chroma
src/retrieval.py               citas por número, reescritura, BM25 + denso, reranker → top_k
src/reranker.py                cross-encoder local (transformers + torch)
src/rag_chain.py               prompt y generación con citas
src/check_models.py            validación de clave, modelos, reranker e índice
src/evaluate.py                evaluación con RAGAS
src/eval_retrieval.py          evaluación de la recuperación con preguntas cotidianas (sin juez)
src/eval_fuera_de_alcance.py   preguntas que el Código no contesta
src/bench_latency.py           latencia por etapas
tests/test_parser.py           tests del parser del BOE
tests/test_tokenizer.py        tests del tokenizador de BM25 y de la detección de artículos citados
eval/qa_codigo_civil.jsonl     17 preguntas jurídicas con respuesta de referencia (RAGAS)
eval/qa_cotidiano.jsonl        25 preguntas cotidianas (dev/test) y 5 de citas, con sus artículos
eval/results_codigo_civil.csv  resultados de RAGAS por pregunta
eval/retrieval_cotidiano.csv   resultados de recuperación por pregunta
eval/fuera_de_alcance.json     respuestas a las preguntas fuera de alcance
eval/latency.csv               tiempos por etapa y pregunta
requirements.txt               dependencias principales
requirements-rerank.txt        dependencias del reranker local (versiones antiguas fijadas)
LICENSE                        licencia MIT
```

El texto del Código Civil procede del BOE ([legislación consolidada](https://www.boe.es/buscar/act.php?id=BOE-A-1889-4763)), fuente oficial del texto consolidado.

## Licencia

El código se distribuye bajo licencia [MIT](LICENSE). El texto del Código Civil es una norma publicada en el BOE y no forma parte del repositorio: se descarga al ejecutar la ingesta.
