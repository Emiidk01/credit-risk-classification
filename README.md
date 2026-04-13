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

De las 17 variables disponibles, se seleccionaron las **5 más relevantes** para el objetivo del modelo, basándose en su importancia calculada mediante una correlación de Pearson con `default_flag`:

| Feature | Tipo |
|---|---|
| `risk_score` | numérico |
| `credit_score` | numérico |
| `repayment_delay_days` | numérico | 
| `monthly_income` | numérico |
| `missed_payments` | numérico |

Por lo tanto, variables como `user_id`, `transaction_date` y `location` fueron descartadas por no aportar valor predictivo al modelo.

### Separación del Set de Entrenamiento y Prueba

Con los datos seleccionados, se realizó la división del dataset en dos subconjuntos con una proporción **80/20**:
- **`X_train` / `y_train`**: utilizados para entrenar el modelo (8,276 registros)
- **`X_test` / `y_test`**: utilizados para evaluar su desempeño sobre datos no vistos (2,069 registros)

### Preprocesado del Target

La variable objetivo `default_flag` ya se encuentra en formato binario entero (`0` y `1`), por lo que **no requiere Label Encoding**.

### Escalamiento de Features: Standardization

Las features seleccionadas presentan rangos de magnitud muy distintos entre sí. Por ejemplo, `monthly_income` alcanza valores de hasta 145,000 mientras que `missed_payments` oscila entre 0 y 7. Esta diferencia no refleja importancia, sino únicamente la unidad de medida de cada variable.

Para eliminar este sesgo, se aplicó **Standardization**, que transforma cada valor según la fórmula:

```
z = (x - μ) / σ
```

Esto garantiza que todas las features tengan **media 0 y desviación estándar 1**, permitiendo que el modelo evalúe cada variable en igualdad de condiciones. La media y desviación estándar se calcularon **únicamente con el set de entrenamiento** y se aplicaron al set de prueba, evitando cualquier fuga de información hacia los datos de evaluación.
