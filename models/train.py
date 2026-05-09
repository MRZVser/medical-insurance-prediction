import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#ЭТАП 1 - Подготовка данных
print("ЭТАП 1: Подготовка данных")
data_path = "data/insurance.csv"

try:
    df = pd.read_csv(data_path)
    print(f"Данные успешно загружены!")
    print(f"Размер датасета: {df.shape}")
except FileNotFoundError:
    print(f"Ошибка: файл не найден -> {data_path}")
    exit()

# Просмотр первых строк
print("\nПервые 5 строк датасета:")
print(df.head())

# 2. Определение признаков и целевой переменной

"""
ПРИЗНАКИ (X):
- age        -> возраст
- sex        -> пол
- bmi        -> индекс массы тела
- children   -> количество детей
- smoker     -> курит ли человек
- region     -> регион проживания

ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (y):
- charges -> стоимость медицинской страховки

"""
# Разделение данных
X = df.drop(columns=["charges"])
y = df["charges"]


# 3. Преобразование категеориальных признаков
print("\nКодирование категориальных признаков")

# Пол:
X["sex"] = X["sex"].map({
    "female": 0,
    "male": 1
})

# Курение:
X["smoker"] = X["smoker"].map({
    "no": 0,
    "yes": 1
})

# Регион:
# Преобразуем в несколько бинарных столбцов
X = pd.get_dummies(
    X,
    columns=["region"],
    drop_first=True
)

print("Категориальные признаки успешно преобразованы.")

# 4. Разделение на train / test
print("\n Разделение данных")

"""
80% данных используются для обучения модели,
а 20% — для проверки качества модели
на новых данных, которые модель ранее не видела.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# 5. Масштабирование
print("\nМасштабирование данных")

scaler = StandardScaler()

# Обучаем scaler только на train
X_train_scaled = scaler.fit_transform(X_train)

# Для test используем transform
# без повторного обучения scaler
X_test_scaled = scaler.transform(X_test)

print("Масштабирование успешно выполнено.")

print("\nДанные полностью подготовлены!")
print(f"Train objects: {X_train_scaled.shape[0]}")
print(f"Test objects: {X_test_scaled.shape[0]}")

#ЭТАП 2 - Обучение и диагностика
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("\nЭТАП 2: Обучение и диагностика")

# 1. Собственная модель(простое правило)
# Обучение: считаем среднее значение целевой переменной для курящих и некурящих
# Мы делаем это на исходном X_train, так как там smoker — это 0 и 1
mean_smoker = y_train[X_train["smoker"] == 1].mean()
mean_non_smoker = y_train[X_train["smoker"] == 0].mean()

def simple_predict(X_input):
    """
    Простейший алгоритм: предсказывает среднюю стоимость 
    только на основе факта курения.
    """
    return X_input["smoker"].apply(lambda x: mean_smoker if x == 1 else mean_non_smoker)

# 2. Готовый алгоритм (Linear Regression)
ml_model = LinearRegression()
# Обучаем на отмасштабированных данных
ml_model.fit(X_train_scaled, y_train)

# 3. Кросс-валидация
# Используем neg_mean_absolute_error, чтобы получить MAE в цикле кросс-валидации
cv_mae_scores = cross_val_score(ml_model, X_train_scaled, y_train, cv=5,scoring='neg_mean_absolute_error')
print(f"Средний MAE на кросс-валидации (5 фолдов): {-cv_mae_scores.mean():.2f}$")

# 4. Расчет метрик
# Получаем предсказания на тестовой выборке
y_pred_simple = simple_predict(X_test)
y_pred_ml = ml_model.predict(X_test_scaled)

def evaluate_models(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) # Корень из MSE для понимания ошибки в долларах
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- Метрики для: {name} ---")
    print(f"MAE (Средняя абсолютная ошибка): {mae:.2f}$")
    print(f"RMSE (Среднеквадратичная ошибка): {rmse:.2f}$")
    print(f"R2 (Коэффициент детерминации): {r2:.3f}")

evaluate_models("Simple Rule (Smoker)", y_test, y_pred_simple)
evaluate_models("Linear Regression ", y_test, y_pred_ml)


# 5. Визуализация и анализ ошибок
# Таблица ТОП-5 самых больших ошибок (где модель промахнулась сильнее всего)
analysis_df = pd.DataFrame({
    'Реальная цена': y_test,
    'Предсказание ML': y_pred_ml,
    'Абс. Ошибка ($)': abs(y_test - y_pred_ml)
})

print("\nТОП-5 самых больших ошибок модели:")
print(analysis_df.sort_values(by='Абс. Ошибка ($)', ascending=False).head(5))

# График: Реальность vs Предсказание
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ml, alpha=0.6, color='royalblue', label='Предсказания Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальный прогноз')

plt.title('Диагностика ошибок: Реальные значения vs Предсказания')
plt.xlabel('Реальная стоимость ($)')
plt.ylabel('Предсказанная стоимость ($)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#ЭТАП 3 - Финальный отбор и сохранение
import os
import joblib

print("\nЭТАП 3: Финальный отбор и сохранение модели")

# 1. Выбор лучшей модели
# По результатам сравнения Linear Regression показала лучшие метрики (MAE, RMSE, R2).
best_model = ml_model
best_model_name = "Linear Regression"

# 2. Финальная проверка на отложенных данных
final_predictions = best_model.predict(X_test_scaled)

final_mae = mean_absolute_error(y_test, final_predictions)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
final_r2 = r2_score(y_test, final_predictions)

print(f"\nФинальные метрики модели {best_model_name}:")
print(f"MAE: {final_mae:.2f}$")
print(f"RMSE: {final_rmse:.2f}$")
print(f"R2: {final_r2:.3f}")

# 3. Создание папки models (если её нет)
os.makedirs("models", exist_ok=True)

# 4. Сохранение в формате .pkl через joblib
model_path = "models/linear_regression_model.pkl"
scaler_path = "models/scaler.pkl"

joblib.dump(best_model, model_path)
print(f"\nМодель сохранена: {model_path}")

joblib.dump(scaler, scaler_path)
print(f"Scaler сохранена: {scaler_path}")

# 5. Итоговый отчет
print("ИТОГОВЫЙ ОТЧЕТ")
print(f"Лучшая модель: {best_model_name}")
print(f"""
Ключевые метрики на новых данных:
- MAE: {final_mae:.2f}$
- RMSE: {final_rmse:.2f}$
- R2: {final_r2:.3f}
""")

print("""
Анализ ошибок:
Модель чаще всего ошибается на очень дорогих страховых случаях,
где стоимость значительно выше средней.
Линейная регрессия имеет тенденцию занижать прогноз
для клиентов с высокой стоимостью страховки.
""")
