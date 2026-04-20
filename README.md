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
| Paid (0) | 0.72 | 0.86 | 0.78 |
| Default (1) | 0.67 | 0.45 | 0.54 |
 
La **matriz de confusión** se incluye como herramienta de interpretación visual, diferenciando entre verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

<img width="515" height="390" alt="image" src="https://github.com/user-attachments/assets/6772a518-f69f-4350-a727-b241719b88e2" />

 
> Chang, V., Sivakulasingam, S., Wang, H., Wong, S. T., Ganatra, M. A., & Luo, J. (2024). Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers. *Risks*, 12(11), 174. https://doi.org/10.3390/risks12110174
 
> Akinjole, A., Shobayo, O., Popoola, J., Okoyeigbo, O., & Ogunleye, B. (2024). Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction. *Mathematics*, 12(21), 3423. https://doi.org/10.3390/math12213423
