# Лабораторная работа 4
Машинное обучение

Научиться работать с библиотекой natasha. Задачи:
i.	Загрузить текстовые данные (не менее 2000 символов)
ii.	Разделить текст на предложения
iii.	Выделить токены и провести частеречную разметку, вывести на экран первые 20 токенов с тэгами
iv.	Нормализовать именованные сущности в тексте
v.	Выделить даты и вывести их в формате число-месяц-год


Средствами NLTK выделить именованные сущности с тэгами (Person, Organisation, GSP и проч.) для английского и русского текста.  
Пример вывода для английского текста:
(PERSON Alse/NNP Young/NNP)
(GPE America/NNP)
(ORGANIZATION Hartford/NNP)
(ORGANIZATION Old/NNP)
(ORGANIZATION House/NNP)
(GSP Connecticut/NNP)
(PERSON Young/NNP)

С помощью sklearn обучить модель распознавать части речи в предложении. Для этого необходимо разбить данные на обучающую и тестовую выборки, а в конце вывести на экран предсказание модели и степень его точности.
Пример ввода: 
X_train = ["This is a positive text", "This is a negative text", "This is a neutral text"]
y_train = ["positive", "negative", "neutral"]
X_test = ["This is a positive text", "This is a negative text"]
y_test = ['positive', 'negative']

Пример вывода:
['positive' 'negative']
Accuracy:  1.0

Необходимые импорты:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


Choco Доставка - это сервис доставки посылок из магазинов до клиентов. Пример клиентов: Kaspi магазин, Алсер, Europharma, KFC и другие.
У Чоко Доставки есть проблема - дисбаланс заказов и курьеров. При повышенном спросе у компании больше заказов, чем курьеров, из-за чего увеличивается время доставки, так как курьеру нужно освободиться, чтобы взять новый заказ. Когда курьеров больше, чем заказов, компания начинает им платить за простой.
Часть команды считает, что наймом бОльшего количества курьеров можно решить проблему, однако с этим согласны не все. Мы хотим проверить гипотезу, что прогнозирование спроса заказов позволит минимизировать долгое время доставки / простой курьеров.
Допустим,  что распределение заказов Алматы схоже с распределением поездок в Чикаго в Соединенных Штатах. Проведем наше исследование на данных по Чикаго.
Необходимые данные, таблицы и датасеты находятся в этой папке: https://drive.google.com/drive/u/0/folders/1l2ns0oLM4Of6rJSKXNMIIx4Oy9SrSmLi

Гипотеза:

Прогнозирование спроса заказов позволит оптимизировать работу курьеров и повысит скорость доставки без найма дополнительных курьеров.

Цель:

Повысить эффективность доставки, не нанимая дополнительных курьеров.

Задачи:

1. Провести анализ данных (прочитать таблицу и визуализировать shapefile).
2. Преобразовать и обогатить данные для обучения модели.
	Interval Hourly -> в формат datetime
	Создать столбец, отображающий дни недели в числовом формате
	Создать столбец, отображающий выходной/будний день
	Создать столбец, отображающий дату
	Создать столбец, отображающий час (0-23)
Создать столбец, отображающий Время суток
Найти минимум и максимум поездок по GEOID, типу дня и времени суток
Визуализировать спрос на поездки для разных GEOID, объединив данные с shapefile
*3. Построить модель прогноза спроса поездок.
	Необходимые импорты:
	import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
4. Протестировать модель.
5. Сохранить модель.
6. Подвести итоги.

Необходимые импорты:
import geopandas as gpd
import missingno as msgn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
