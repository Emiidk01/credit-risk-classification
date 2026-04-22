# Clasificador de Riesgo de Default en Buy Now Pay Later (BNPL)

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de **machine learning** capaz de predecir si un usuario de un servicio **Buy Now Pay Later (BNPL)** va a incumplir su pago (**default**) o va a pagarlo correctamente. La meta es identificar, de forma anticipada, qué perfiles de usuarios representan un riesgo crediticio alto.

### Contexto

BNPL es un modelo de financiamiento a corto plazo que permite a los consumidores adquirir un producto de inmediato y pagarlo en cuotas posteriores. El proveedor asume el riesgo de impago, por lo que contar con un clasificador preciso es crítico para la sostenibilidad del negocio. Este modelo busca apoyar la toma de decisiones al momento de aprobar o rechazar una transacción, enfocando los recursos en perfiles con alta probabilidad de cumplimiento.

## Descripción del Dataset

Para este proyecto se utiliza el dataset [**Buy Now Pay Later BNPL Credit Risk Dataset**](https://www.kaggle.com/datasets/shree0910/buy-now-and-pay-later-fintech-ml-dataset/data), disponible públicamente en **Kaggle**. Este conjunto de datos contiene **10,345 observaciones** de transacciones BNPL con un total de **17 variables** descriptivas.

La **variable objetivo (`default_flag`)** es de tipo binaria:
- `0` → el usuario pagó correctamente *(Paid)*
- `1` → el usuario incumplió el pago *(Defaulted)*

De los 10,345 registros, **6,305 corresponden a usuarios que pagaron (60.9%)** y **4,040 a usuarios que defaultearon (39.1%)**.

## Proceso

### Selección de Features

Para determinar qué variables incluir en el modelo se calculó la
**correlación de Pearson** entre cada variable numérica del dataset y la
variable objetivo `default_flag`. Se estableció un umbral mínimo de
correlación absoluta de **0.10**, seleccionando únicamente las features
que superan ese valor, lo que indica una relación lineal significativa
con el target.

Las **features seleccionadas** son:

| Feature | Tipo | Descripción | Correlación absoluta |
|---|---|---|---|
| `risk_score` | numerico | Score de riesgo calculado del usuario (0–398) Más alto = mayor riesgo | 0.40 |
| `credit_score` | 	numérico | Puntaje crediticio estándar (300–850) | 0.32 |
| `repayment_delay_days` | 	numérico | Días de retraso en pagos anteriores (0–33) | 0.28 |
| `monthly_income` | 	numérico | Ingreso mensual del usuario en USD | 0.27 |
| `missed_payments` | 	numérico | Número de pagos omitidos (0-7) | 0.27 |
| `debt_to_income_ratio` | numérico | Razón deuda / ingreso mensual | 0.17 |

Variables como `bnpl_installments` (0.01), `app_usage_frequency` (0.004)
y `location` fueron descartadas por no superar el umbral, indicando que
no tienen una relación lineal significativa.
Variables no numéricas como `user_id`, `transaction_date` y `location` fueron descartadas por no aportar valor predictivo al modelo.

<img width="889" height="489" alt="Correlación absoluta con default_flag: criterio de selección de features" src="https://github.com/user-attachments/assets/c643277f-7a14-4187-a2b3-6ed2edf5bfd7" />

### Separación del Set de Entrenamiento y Prueba

Con los datos seleccionados, se realizó la división del dataset en dos subconjuntos con una proporción **80/20**:
- **`X_train` / `y_train`**: utilizados para entrenar el modelo (8,276 registros)
- **`X_test` / `y_test`**: utilizados para evaluar su desempeño sobre datos no vistos (2,069 registros)

### Preprocesado del Target

La variable objetivo `default_flag` ya se encuentra en formato binario entero (`0` y `1`), por lo que **no requiere Label Encoding**.

### Escalamiento de Features: Standardization

Las features seleccionadas presentan rangos de magnitud muy distintos entre sí. Por ejemplo, `monthly_income` alcanza valores de hasta 145,000 mientras que `missed_payments` oscila entre 0 y 7. Esta diferencia no refleja importancia, sino únicamente la unidad de medida de cada variable.

Para eliminar este sesgo, se aplicó **Standardization**, que transforma cada valor según la fórmula:


$$z = \frac{x - \mu}{\sigma}$$


Esto garantiza que todas las features tengan **media 0 y desviación estándar 1**, permitiendo que el modelo evalúe cada variable en igualdad de condiciones. La media y desviación estándar se calcularon **únicamente con el set de entrenamiento** y se aplicaron al set de prueba, evitando cualquier fuga de información hacia los datos de evaluación.

## Implementación del Modelo
 
### Modelo seleccionado: Multilayer Perceptron (MLP)
 
Se implementó un **Multilayer Perceptron (MLP)** con dos capas ocultas para la tarea de clasificación binaria de riesgo de default. La elección de este modelo está respaldada por Yakubu et al. (2025), quienes demuestran que una arquitectura MLP relativamente simple, cuando se combina con estandarización de datos mediante Z-score, puede superar modelos más complejos como Deep Neural Networks (DNNs), obteniendo mejoras sustanciales en métricas críticas como Recall y F1-Score.
 
### Arquitectura
 
| Capa | Tipo | Neuronas | Activación |
|---|---|---|---|
| Entrada | Input | 6 | — |
| Oculta 1 | Dense | 64 | ReLU |
| Oculta 2 | Dense | 32 | ReLU |
| Salida | Dense | 1 | Sigmoid |
 
- **Función de pérdida:** `binary_crossentropy`, estándar para clasificación binaria
- **Optimizador:** `Adam`, adaptativo y eficiente para datos tabulares
- **Épocas:** 50

La activación **ReLU** en las capas ocultas permite aprender relaciones no lineales entre las features. La activación **Sigmoid** en la capa de salida produce una probabilidad entre 0 y 1, interpretada como la probabilidad de default del usuario.
 
> Yakubu, R., Abubakar, A. A., Lazarus, D. G., & Babajo, A. A. (2025). Enhanced Credit Card Default Prediction Using a Multilayer Perceptron With Z-Score–Based Outlier Handling on The UCI Dataset. *Researchers Journal of Science and Technology*, 5(8), 18–30. https://www.rejost.com.ng/index.php/home/article/view/230
 
## Evaluación del Modelo
 
### Métricas seleccionadas
 
Para evaluar el desempeño del modelo se utilizaron **Accuracy, Precision, Recall y F1-Score**, métricas ampliamente adoptadas en la literatura de credit risk. Chang et al. (2024) emplean este mismo conjunto de métricas para comparar modelos de clasificación de riesgo crediticio, incluyendo redes neuronales, regresión logística y modelos de boosting, concluyendo que constituyen el estándar para evaluar la eficiencia de algoritmos en este dominio.
 
Adicionalmente, Akinjole et al. (2024) refuerzan la importancia de reportar Precision y Recall de forma conjunta en problemas de default, dado que los datasets de crédito suelen presentar desbalance de clases. Un modelo con alta Accuracy no garantiza un buen desempeño real: un clasificador que prediga siempre "no default" puede alcanzar alta Accuracy pero tendrá un Recall de 0 para la clase positiva, lo que lo hace inútil en la práctica.
 
| Métrica | Descripción |
|---|---|
| **Accuracy** | Proporción de predicciones correctas sobre el total |
| **Precision** | De los predichos como default, ¿cuántos realmente lo fueron? |
| **Recall** | De todos los defaults reales, ¿cuántos detectó el modelo? |
| **F1-Score** | Media armónica entre Precision y Recall |

| Clase | Precision | Recall | F1-Score |
|---|---|---|---|
| Paid (0) | 0.70 | 0.92 | 0.80 |
| Default (1) | 0.76 | 0.40 | 0.53 |
| **Accuracy** | | | **0.71** |
 
La **matriz de confusión** se incluye como herramienta de interpretación visual, diferenciando entre verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

<img width="515" height="390" alt="image" src="https://github.com/user-attachments/assets/6772a518-f69f-4350-a727-b241719b88e2" />

 
> Chang, V., Sivakulasingam, S., Wang, H., Wong, S. T., Ganatra, M. A., & Luo, J. (2024). Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers. *Risks*, 12(11), 174. https://doi.org/10.3390/risks12110174
 
> Akinjole, A., Shobayo, O., Popoola, J., Okoyeigbo, O., & Ogunleye, B. (2024). Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction. *Mathematics*, 12(21), 3423. https://doi.org/10.3390/math12213423

## Refinamiento del Modelo
 
### Diagnóstico del modelo base
 
El modelo v1 presentó un **Recall de 0.40 en la clase Default (1)**, lo que significa que dejaba pasar el 60% de los incumplimientos reales. En un contexto de riesgo crediticio, este es el error más costoso, ya que un default no detectado representa una pérdida directa para el proveedor BNPL.

### Ajustes realizados
 
**1. Class weight balancing**
Se asignaron pesos inversamente proporcionales a la frecuencia de cada clase mediante weighted cross-entropy. Ya que fue una técnica que emplee debido a que  que mis clases estaban ligeramente desbalanceados, cosa que habia notado y que supuse que me daria un problema. El uso de `class_weight` corrige este sesgo penalizando más los errores cometidos sobre defaults durante el entrenamiento, que en este caso es la clase minoritaria.

**2. Dropout (0.3 y 0.2)**
Se agregaron capas de Dropout tras cada capa oculta para reducir el overfitting, forzando al modelo a aprender representaciones más generalizables (Srivastava et al., 2014).
 
**3. EarlyStopping**
Se reemplazaron las 50 épocas fijas por entrenamiento con parada temprana, monitoreando `val_loss` con una paciencia de 10 épocas. Prechelt (2012) establece que detener el entrenamiento en el momento adecuado evita que el modelo memorice los datos de entrenamiento y pierda capacidad de predecir correctamente datos nuevos.

### Resultados — Modelo v2 (refinado)
 
| Clase | Precision | Recall | F1-Score |
|---|---|---|---|
| Paid (0) | 0.79 | 0.57 | 0.66 |
| Default (1) | 0.54 | 0.76 | 0.63 |
| **Accuracy** | | | **0.65** |
 
### Comparación v1 vs v2
 
| Métrica | Modelo v1 | Modelo v2 | Cambio |
|---|---|---|---|
| Accuracy | 0.71 | 0.65 | -0.06 |
| Precision Default (1) | 0.76 | 0.54 | -0.22 |
| **Recall Default (1)** | **0.40** | **0.76** | **+0.36** |
| F1-Score Default (1) | 0.53 | 0.63 | **+0.10** |


<img width="506" height="390" alt="Matriz de confusion modelo 2" src="https://github.com/user-attachments/assets/07b3f27a-eda1-4289-8eb4-4dd068f24cdd" />

### Observaciones finales

El refinamiento estuvo orientado a mejorar la detección de usuarios con riesgo de default, que era la debilidad más crítica del modelo v1. Mediante la incorporación de class weight balancing, Dropout y EarlyStopping, el modelo v2 logró subir el Recall de Default de 0.40 a 0.76, detectando correctamente el 76% de los incumplimientos reales frente al 40% del modelo anterior. El F1-Score de Default también mejoró de 0.53 a 0.63.
 
Este avance tiene un costo: la Accuracy general bajó de 0.71 a 0.65 y la Precision de Default disminuyó de 0.76 a 0.54, lo que significa que el modelo v2 genera más falsas alarmas. Sin embargo, en un contexto de riesgo crediticio este trade-off es aceptable, ya que el costo de no detectar un default real es mayor que el costo de rechazar a alguien que si podría pagar.

> Prechelt, L. (2012). Early Stopping — But When? In: Neural Networks: Tricks of the Trade. Lecture Notes in Computer Science, vol 7700. Springer. https://doi.org/10.1007/978-3-642-35289-8_5
 
> Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.
