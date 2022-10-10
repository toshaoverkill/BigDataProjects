import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

HEADERS = 'netflix1.csv'
EXITHEADER = 'Введите номер задания или введите \'0\' для выхода:'
KProjects = pd.read_csv(HEADERS)


def ex1(data=KProjects):
    print(data)


def ex2(data=KProjects):
    print('info():')
    data.info()
    print('head():\n', data.head())
    print('Пропуски:\n', data.isna().sum())


def ex3(data=KProjects):
    B = data['release_year'].value_counts()
    A = B[B.index > 1995]
    print('Выбор варианта вывода графика (1 или 2):')
    a = int(input())
    match a:
        case 1:
            fig = go.Figure(px.bar(x=A.index, y=A.values, color=A.values, text=A.values))
            fig.update_traces(textfont_size=16, textangle=0, textposition='outside',
                              marker=dict(line=dict(color='black', width=1)))
            fig.update_layout(
                title='Диаграмма количества сериалов и ТВ шоу Netflix за последние 26 лет', title_font_size=20,
                title_x=0.5,
                xaxis_title='Года', xaxis_title_font_size=14, xaxis_tickfont_size=14,
                yaxis_title='Кол-во', yaxis_title_font_size=14, yaxis_tickfont_size=14,
                yaxis=dict(dtick=100),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            fig.update_xaxes(tickangle=315, tickfont_size=14, automargin=True)
            fig.update_yaxes(tickfont_size=14, automargin=True)
            fig.show()
        case 2:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[1].bar(A.index, A.values, color='teal', edgecolor='black')
            ax[1].grid(alpha=0.3, zorder=1)
            ax[1].set_title('Диаграмма количества сериалов и ТВ\nшоу Netflix за последние 26 лет')
            ax[1].set_ylabel('Кол-во')
            ax[0].barh(A.index, A.values, color='teal', edgecolor='black')
            ax[0].grid(alpha=0.3, zorder=1)
            ax[0].set_title('Диаграмма количества сериалов и ТВ\nшоу Netflix за последние 26 лет')
            ax[0].set_ylabel('Года')
            plt.show()


def ex4(data=KProjects):
    B = data['release_year'].value_counts()
    print('Выбор варианта вывода графика (1 или 2):')
    a = int(input())
    match a:
        case 1:
            A = B[B.index > 1995]
            fig = go.Figure()
            fig.add_trace(go.Pie(values=A.values, labels=A.index))
            fig.update_layout(title='Диаграма соотношения количества сериалов и ТВ шоу на Netflix',
                              title_y=0.96, title_x=0.55, title_xanchor='center', title_yanchor='top',
                              title_font_size=20)
            fig.show()
        case 2:
            A = B[B.index > 2012]
            fig, ax = plt.subplots()
            explode_val = (0.1, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.2)
            ax.pie(A.values, labels=A.index, autopct='%1.1f%%', explode=explode_val)
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            ax.set_title("Диаграма соотношения количества сериалов и ТВ шоу на Netflix")
            plt.show()


def ex5(data=KProjects):
    B = data['release_year'].value_counts()
    A = B[B.index > 1995]
    print('Выбор варианта вывода графика (1 или 2):')
    a = int(input())
    match a:
        case 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=A.sort_index().index, y=A.sort_index().values, mode='lines+markers',
                                     marker=dict(color='darkblue', size=10, line=dict(color='black', width=3))))
            fig.update_layout(title='Динамика роста количества сериалов и ТВ шоу на Netflix',
                              title_y=0.96, title_x=0.55, title_xanchor='center', title_yanchor='top',
                              title_font_size=20,
                              xaxis_title='Год', xaxis_title_font_size=16,
                              yaxis_title='Количество', yaxis_title_font_size=16,
                              width=None, height=750)
            fig.update_traces(line_color='crimson', selector=dict(type='scatter')
                              )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='azure')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='azure')
            fig['data'][0]['showlegend'] = True
            fig['data'][0]['name'] = 'Динамика'
            fig.show()
        case 2:
            plt.figure(figsize=(12, 7))
            plt.plot(A.sort_index().index, A.sort_index().values, 'o-r', alpha=0.7, label="Динамика", lw=5, mec='b',
                     mew=2, ms=10)
            plt.legend()
            plt.grid(True)
            plt.show()


print('Введите номер задания:')
example = int(input())
while True:
    match example:
        case 1:
            ex1()
            print(EXITHEADER)
            example = int(input())
        case 2:
            ex2()
            print(EXITHEADER)
            example = int(input())
        case 3:
            ex3()
            print(EXITHEADER)
            example = int(input())
        case 4:
            ex4()
            print(EXITHEADER)
            example = int(input())
        case 5:
            ex5()
            print(EXITHEADER)
            example = int(input())
        case 0:
            print("Выход")
            break
