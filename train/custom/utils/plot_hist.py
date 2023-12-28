
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np

def plot_hist(df):
    fig = go.Figure()

    fontsize_small = 20
    fontsize_big = 25
    font_family = "Times New Roman"

    color_discrete_map = {'Time(ms)': 'rgb(50, 100, 200)', 'Para(M)': 'rgb(200, 100, 90)'}
    df_melt = df.melt(id_vars='compare_models', var_name=' ', value_name='value')
    fig = px.bar(df_melt, x='compare_models', y='value', color=' ', barmode='group',color_discrete_map=color_discrete_map)
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont=dict(family=font_family, size=fontsize_small))


    layout = go.Layout(
        title=' 测试时间和模型大小比较',
        title_x=0.5,
        margin=dict(
            b=0,  
            l=5,
        ),
        legend=dict(
            font=dict(
                family=font_family,
                size=fontsize_small
                ),
            x=0.01,  # 控制legend的x位置
            y=1.03,  # 控制legend的y位置
            bgcolor='rgba(0, 0, 0, 0)',  # 设置图例的背景颜色为透明,
        ),
        width=1200,  # 设置图表的宽度
        height=620,  # 设置图表的高度
        plot_bgcolor='rgba(0, 0, 0, 0)',
        titlefont=dict(size=fontsize_big, family=font_family),
    )
    fig.update_layout(layout)
    
    fig.update_xaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='lightgrey', title_text="",tickfont=dict(size=fontsize_small,family=font_family), titlefont=dict(size=fontsize_small, family=font_family))
    fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, gridwidth=1, gridcolor='lightgrey',title_text='Value', tickfont=dict(size=fontsize_small,family=font_family),tickformat=".0f", titlefont=dict(size=fontsize_small, family=font_family), range=[0, 210], tickmode='array', tickvals=list(np.arange(0, 210, 10)))
    
    fig.show()

if __name__ == "__main__":
    data = {
        'compare_models': ["ResUnet", "MWResUnet","Cascade_ResUnet","DnCNN","MWCNN","Restormer","SwinIR"],
        'Time(ms)': [8, 15, 30, 16, 10, 126, 130],
        'Para(M)': [56, 97, 56, 2, 24, 99, 195],
        }
    df = pd.DataFrame(data)

    # 调用 plot_hist 函数
    plot_hist(df)

