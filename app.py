import gradio as gr
import pandas as pd
import joblib

# Загрузка модели и скалера
model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# функция предсказания
def predict_insurance(age, sex, bmi, children, smoker, region):
    # Защита от "дурака"
    try:
        age = int(age)
        bmi = float(bmi)
        children = int(children)

    except:
        return "Ошибка: возраст, BMI и количество детей должны быть числами."

    # Проверка возраста
    if not (18 <= age <= 80):
        return "Ошибка: возраст должен быть в диапазоне 18–80 лет."

    # Проверка BMI
    if not (10 <= bmi <= 60):
        return "Ошибка: BMI должен быть в диапазоне 10–60."

    # Проверка количества детей
    if not (0 <= children <= 10):
        return "Ошибка: количество детей должно быть от 0 до 10."

    # Проверка пола
    if sex not in ["male", "female"]:
        return "Ошибка: пол должен быть male или female."

    # Проверка курения
    if smoker not in ["yes", "no"]:
        return "Ошибка: smoker должен быть yes или no."

    # Проверка региона
    valid_regions = [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]

    if region not in valid_regions:
        return "Ошибка: неверное значение региона."

    # Кодирование категориальных признаков

    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0

    # One-Hot Encoding региона
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    # northeast -> все нули

    # Создание DF
    # ВАЖНО: порядок колонок должен совпадать
    # с порядком при обучении модели
    input_data = pd.DataFrame([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region_northwest,
        region_southeast,
        region_southwest
    ]], columns=[
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region_northwest",
        "region_southeast",
        "region_southwest"
    ])

    # МАСШТАБИРОВАНИЕ

    input_scaled = scaler.transform(input_data)

    print("\nSCALED DATA:")
    print(input_scaled)

    # ПРЕДСКАЗАНИЕ

    prediction = model.predict(input_scaled)[0]

    print("\nPREDICTION:")
    print(prediction)


    # Защита от отрицательного прогноза
    prediction = max(prediction, 0)
    # Возврат результата
    return f"Прогнозируемая стоимость страховки: ${prediction:,.2f}"

# Создание интерфейса 
inputs = [

    gr.Number(label="Возраст", value=25),#создает числовое поле ввода

    gr.Radio(choices=["male", "female"],label="Пол",value="male"),#создает выбор одного варианта.

    gr.Number(label="BMI",value=25.0),

    gr.Number(label="Количество детей",value=0),

    gr.Radio(choices=["yes", "no"], label="Курит?",value="no"),

    gr.Dropdown(#создает выпадающий список
        choices=[
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ],
        label="Регион",
        value="northeast"
    )
]
output = gr.Textbox(#Поле, куда выводится ответ модели
    label="Результат модели"
)
# Интерфейс
#gr.Interface() связывает все воедино:fn=predict_insurance - говорим какую функцию вызвать когда пользователь нажмет submit на сайте
#inputs=inputs и outputs=output: Мы передаем созданные ранее списки элементов.
#title и description: Текст, который отображается в самом верху страницы.Это делает приложение понятным для пользователя.
interface = gr.Interface(fn=predict_insurance, inputs=inputs, outputs=output,

    title="Прогноз стоимости медицинской страховки",

    description="""
    Введите данные пациента,
    и модель машинного обучения предскажет
    примерную стоимость медицинской страховки.
    """
)
# Запуск
if __name__ == "__main__":#Запускать сервер только если файл запущен напрямую

    print("Интерфейс готов!")
    interface.launch(share=True,theme="soft")#launch запускает веб сервер и ждет ввода
                                            #share = True: Создаёт публичную ссылку
                                            # theme="soft" : Тема оформления интерфейса.

