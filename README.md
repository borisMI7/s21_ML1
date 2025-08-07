# ML-Intro: Прогнозирование стоимости аренды жилья

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-1.3.5-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.23.1-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

Этот проект является введением в машинное обучение и выполнен в рамках учебной программы [Школы 21](https://21-school.ru/). Основная цель — пройти все базовые этапы построения ML-модели: от первичного анализа данных до обучения нескольких простых моделей и сравнения их эффективности.

## Цель проекта

Главная задача — разработать регрессионную модель для прогнозирования стоимости аренды квартиры на основе данных с сайта renthop.com. В ходе проекта мы анализируем такие признаки, как количество ванных комнат, спален и уровень интереса к объявлению.

## Данные

Данные взяты из соревнования на Kaggle: [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data). Для анализа используется файл `train.json`.

## Этапы работы

Проект разделен на несколько ключевых этапов:

1.  **Теоретическая часть:**
    *   Ответы на вопросы о применении ML, разнице между типами задач (классификация, регрессия, мультикласс, мультилейбл).

2.  **Исследовательский анализ данных (EDA):**
    *   Загрузка и первичный осмотр данных (`.info()`, `.describe()`).
    *   Анализ целевой переменной (`price`): построение гистограмм и boxplot-диаграмм для выявления распределения и выбросов.
    *   Очистка данных: удаление выбросов в цене на основе 1 и 99 перцентилей.
    *   Анализ признаков: кодирование категориального признака `interest_level`.

3.  **Анализ зависимостей:**
    *   Построение матрицы корреляций и тепловой карты для оценки взаимосвязи между признаками и целевой переменной.
    *   Визуализация зависимостей с помощью диаграмм рассеяния (scatterplot).

4.  **Инжиниринг признаков (Feature Engineering):**
    *   Создание полиномиальных признаков с помощью `sklearn.preprocessing.PolynomialFeatures` для усложнения модели.

5.  **Обучение и оценка моделей:**
    *   Данные были разделены на обучающую и тестовую выборки.
    *   Были обучены и оценены следующие модели:
        *   **Linear Regression** (Линейная регрессия)
        *   **Decision Tree Regressor** (Дерево решений)
        *   **Наивные модели** (прогноз на основе среднего и медианного значения)

## Результаты

Для оценки качества моделей использовались метрики **MAE** (Mean Absolute Error) и **RMSE** (Root Mean Square Error). Сравнение производительности моделей и подробный анализ результатов находятся непосредственно в Jupyter-ноутбуке `ml1.ipynb`.

## Как запустить проект

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name
    ```

2.  **Создайте и активируйте виртуальное окружение (рекомендуется):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # для Windows: venv\Scripts\activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install jupyter pandas numpy scikit-learn matplotlib seaborn scipy statsmodels lightgbm
    ```

4.  **Подготовьте данные:**
    *   Скачайте файл `train.json` со страницы соревнования [Kaggle](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data).
    *   Создайте в корне проекта папку `data` и поместите в нее скачанный файл.

5.  **Запустите Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Откройте и выполните ячейки в файле `ml1.ipynb`.

## Используемые технологии

*   Python 3
*   Jupyter Notebook
*   Pandas & NumPy
*   Scikit-learn
*   Matplotlib & Seaborn
*   SciPy & Statsmodels
*   LightGBM