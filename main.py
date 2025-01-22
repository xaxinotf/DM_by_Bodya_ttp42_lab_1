"""
Market Basket Analysis Using Apriori Algorithm with Dash & Plotly
"""

# ---------------------------------------------
# 1. Імпорт необхідних бібліотек
# ---------------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------
# 2. Завантаження та попередня обробка даних
# ---------------------------------------------
# Завантаження даних
file_path = "Groceries_dataset.csv"  # Вкажіть шлях до вашого файлу
groceries_df = pd.read_csv(file_path)

# Перевірка на пропущені значення
missing_values = groceries_df.isnull().sum()
if missing_values.any():
    print("\nData Cleaning: Found missing values in the dataset.")
    print(f"Missing values in each column:\n{missing_values}")
    # Обробка пропущених значень (можна заповнити або видалити)
    # groceries_df.dropna(inplace=True)  # або
    # groceries_df.fillna(method='ffill', inplace=True)  # для заповнення
else:
    print("\nData Cleaning: No missing values found.")

# Перетворення колонок
groceries_df['Date'] = pd.to_datetime(groceries_df['Date'], format='%d-%m-%Y')
groceries_df['itemDescription'] = groceries_df['itemDescription'].str.strip().str.lower()
groceries_df['Transaction_ID'] = groceries_df.groupby(['Member_number', 'Date']).ngroup()

# One-Hot Encoding
basket = groceries_df.groupby(['Transaction_ID', 'itemDescription'])['itemDescription'] \
                     .count() \
                     .unstack() \
                     .fillna(0) \
                     .astype(bool)

# ---------------------------------------------
# 3. Генерація частих наборів товарів і правил
# ---------------------------------------------
# Параметри для Apriori
frequent_itemsets = apriori(basket, min_support=0.003, use_colnames=True)
frequent_itemsets['itemset_length'] = frequent_itemsets['itemsets'].apply(len)

# Генерація асоціативних правил
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)

# Виведення в консоль для частих наборів і правил
print("Frequent Itemsets:")
print(frequent_itemsets.head())

print("\nAssociation Rules:")
print(rules.head())

# Генерація правил асоціацій
print("\nГенерація правил асоціацій:")
print("Генерація корисних правил на основі виявлених наборів елементів, що часто зустрічаються. ")
print("Правила асоціацій, які не досягають порогу в 1, відсікаються. Вище значення підйому означає, що правило є сильнішим/важливішим.")
print("Правила відсортовані в порядку спадання за значеннями достовірності та підйому. ")
print("Чим більші значення довіри та підйому, тим сильніше правило.\n")

# Виведення відсортованих правил
sorted_rules = rules.sort_values(by=['confidence', 'lift'], ascending=False)
print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# ---------------------------------------------
# 4. Створення Dash веб-додатку
# ---------------------------------------------
# Створення Dash додатку
app = dash.Dash(__name__)

# Інтерактивний графік для топ-10 найпопулярніших товарів
item_frequencies = basket.sum().sort_values(ascending=False)

fig = px.bar(
    item_frequencies.head(10),
    x=item_frequencies.head(10).values,
    y=item_frequencies.head(10).index,
    labels={"x": "Frequency", "y": "Items"},
    title="Top 10 Most Frequent Items"
)

# Оновлений графік для Lift vs Confidence
fig_lift_confidence = px.scatter(
    rules,
    x="confidence",
    y="lift",
    title="Lift vs Confidence",
    labels={"confidence": "Confidence", "lift": "Lift"}
)

# Додатковий графік: Розподіл довжини наборів товарів
fig_itemset_length = px.histogram(
    frequent_itemsets,
    x="itemset_length",
    title="Distribution of Itemset Lengths",
    labels={"itemset_length": "Itemset Length"}
)

# Створення веб-сторінки
app.layout = html.Div([
    html.H1("Market Basket Analysis Dashboard"),
    html.Div([
        html.Div([
            html.H3("Top 10 Most Frequent Items"),
            dcc.Graph(id='top-items-graph', figure=fig)
        ], className="six columns"),
        html.Div([
            html.H3("Lift vs Confidence"),
            dcc.Graph(id='lift-confidence-graph', figure=fig_lift_confidence)
        ], className="six columns"),
    ], className="row"),
    html.Div([
        html.Div([
            html.H3("Distribution of Itemset Lengths"),
            dcc.Graph(id='itemset-length-graph', figure=fig_itemset_length)
        ], className="six columns"),
    ], className="row"),
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Оновлення кожні 30 секунд
        n_intervals=0
    )
])

# Оновлення графіка за допомогою періодичних оновлень
@app.callback(
    Output('top-items-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    item_frequencies = basket.sum().sort_values(ascending=False)
    fig = px.bar(
        item_frequencies.head(10),
        x=item_frequencies.head(10).values,
        y=item_frequencies.head(10).index,
        labels={"x": "Frequency", "y": "Items"},
        title="Top 10 Most Frequent Items"
    )
    return fig

@app.callback(
    Output('lift-confidence-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_lift_confidence(n):
    fig_lift_confidence = px.scatter(
        rules,
        x="confidence",
        y="lift",
        title="Lift vs Confidence",
        labels={"confidence": "Confidence", "lift": "Lift"}
    )
    return fig_lift_confidence

@app.callback(
    Output('itemset-length-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_itemset_length(n):
    fig_itemset_length = px.histogram(
        frequent_itemsets,
        x="itemset_length",
        title="Distribution of Itemset Lengths",
        labels={"itemset_length": "Itemset Length"}
    )
    return fig_itemset_length

# Запуск веб-сервера
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
