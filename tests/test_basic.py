import sys
import os

# Добавляем корень проекта в список путей где может лежать файл app
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pytest
from app import predict_insurance, interface


# Тест 1: Проверка работоспособности
def test_prediction_works():

    result = predict_insurance(
        25,"male",25.0,0,"no","northeast"
    )
    assert "Ошибка" not in result
    assert "Прогнозируемая стоимость" in result


# Тест 2: Проверка формата ответа
def test_output_format():

    result = predict_insurance(
        30,"female", 22.0,1,"yes","southwest"
    )
    assert isinstance(result, str)
    assert "$" in result


# Тест 3: Проверка интерфейса
def test_interface_build():

    assert interface.fn is not None
    assert interface.title == "Прогноз стоимости медицинской страховки"
