
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Загрузка данных
data = pd.read_csv('fish_train.csv')  # Укажите путь к вашему файлу


# Разделение данных на обучающую и тестовую выборки
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=11, stratify=data['Species'])

# Вычисление выборочного среднего колонки Width
mean_width = train_data['Width'].mean()
print(f'Выборочное среднее Width: {mean_width:.3f}')  # Округляем до тысячных
# Избавляемся от категориальных признаков
train_data_numeric = train_data.select_dtypes(include='number')
test_data_numeric = test_data.select_dtypes(include='number')

# Обучение модели линейной регрессии
# Замените 'Weight' на имя целевой переменной
X_train = train_data_numeric.drop(columns=['Weight'])
y_train = train_data_numeric['Weight']

model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания и оценка модели
X_test = test_data_numeric.drop(columns=['Weight'])
y_test = test_data_numeric['Weight']
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f'r2_score: {r2:.3f}')  # Округляем до тысячных

# Найдем коррелированные признаки
correlation_matrix = train_data_numeric.corr()
correlated_features = correlation_matrix['Weight'].abs().nlargest(
    4).index.tolist()  # 4, чтобы включить Weight
correlated_features.remove('Weight')  # Удаляем целевую переменную

print(f'Наиболее коррелированные признаки: {
      ", ".join(correlated_features)}')  # Перечисляем признаки
