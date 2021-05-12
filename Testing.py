import math

from flask import Flask,request ,render_template, redirect, url_for

import pandas as pd
import numpy as np
global data

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('basic.html')

@app.route('/info', methods=['POST','GET'])
def info():

    if request.method == 'POST':
       month = request.form['Boston3']
       district = request.form['Boston']
       morning = request.form['Boston1']
       weekend = request.form['Boston2']

       result = main_algorithm(district, weekend, month, morning)
       print('Result is :', result)

       if result == 'Not Dangerous':
           return render_template('ff.html')
       if result == 'Dangerous':
           return render_template('unsafe.html')


@app.route('/graph', methods=['POST','GET'])
def graph():
    return render_template('graph.html', name='Various Graphs')

@app.route('/map1', methods=['POST','GET'])
def map1():
    return render_template('map1.html', name='Top crime', url='static/images/map1.png')

@app.route('/map2', methods=['POST','GET'])
def map2():
    return render_template('map2.html', name='District Crime', url='static/images/map2.png')


def naive_bayes(data_df, weekend, month, dist, morning):
    P_dangerous = 0
    P_not_dangerous = 0
    P_weekend_given_dang = 0
    P_weekend_given_not_dang = 0
    P_morning_given_dang = 0
    P_morning_given_not_dang = 0
    df_real = pd.DataFrame()
    data_means = pd.DataFrame()
    data_var = pd.DataFrame()

    data_df = data_df.loc[:, ['IS_DANGEROUS', 'IS_WEEKEND', 'IS_MORNING', 'DIST_CODES', 'MONTH']]
    n_dangerous = (data_df['IS_DANGEROUS'][data_df['IS_DANGEROUS'] == True]).values.sum()
    n_not_dangerous = len(data_df) - n_dangerous
    total = n_dangerous + n_not_dangerous
    P_dangerous = n_dangerous / total
    P_not_dangerous = n_dangerous / total

    # print('Dangs count ', n_dangerous)
    # print('Not Dangs count ', n_not_dangerous)

    df_real = data_df.loc[:, ['DIST_CODES', 'MONTH', 'IS_DANGEROUS']]
    data_means = df_real.groupby('IS_DANGEROUS').mean()
    data_var = df_real.groupby('IS_DANGEROUS').var()

    dang_dist_mean = data_means['DIST_CODES'][True]
    dang_month_mean = data_means['MONTH'][True]

    dang_dist_var = data_var['DIST_CODES'][True]
    dang_month_var = data_var['MONTH'][True]

    not_dang_dist_mean = data_means['DIST_CODES'][False]
    not_dang_month_mean = data_means['MONTH'][False]

    not_dang_dist_var = data_var['DIST_CODES'][False]
    not_dang_month_var = data_var['MONTH'][False]

    P_weekend_given_dang = data_df['IS_DANGEROUS'][data_df['IS_WEEKEND'] == True].values.sum() / total
    P_weekend_given_not_dang = (data_df['IS_DANGEROUS'][data_df['IS_WEEKEND'] == False]).values.sum() / total

    P_morning_given_dang = data_df['IS_DANGEROUS'][data_df['IS_MORNING'] == True].values.sum() / total
    P_morning_given_not_dang = (data_df['IS_DANGEROUS'][data_df['IS_MORNING'] == False]).values.sum() / total

    # print('Conditional probabilities for IS_WEEKEND and IS_MORNING are as follows \n:')
    # print('P_weekend_given_dang:', P_weekend_given_dang)
    # print('P_weekend_given_not_dang:', P_weekend_given_not_dang)
    # print('P_morning_given_dang:', P_morning_given_dang)
    # print('P_morning_given_not_dang:', P_morning_given_not_dang)
    # print('dang_dist_mean:', dang_dist_mean)
    # print('dang_month_mean:', dang_month_mean)
    # print('dang_dist_var:', dang_dist_var)
    # print('dang_month_var:', dang_month_var)
    #
    # print('not_dang_dist_mean:', not_dang_dist_mean)
    # print('not_dang_month_mean:', not_dang_month_mean)
    # print('not_dang_dist_var:', not_dang_dist_var)
    # print('not_dang_month_var:', not_dang_month_var)

    front_end_is_weekend = weekend
    front_end_is_morning = morning
    front_end_district = dist
    front_end_month = month

    ans_prob_dang = 0
    ans_prob_not_dang = 0
    answer = pd.DataFrame()
    answer['DIST_CODES'] = [3]
    answer['MONTH'] = [10]
    answer_is_weekend = True
    front_end_is_morning = False

    if answer_is_weekend == True:
        prob_weekend_dang = P_weekend_given_dang
        prob_weekend_not_dang = P_weekend_given_not_dang
    else:
        prob_weekend_dang = 1
        prob_weekend_not_dang = 1

    if front_end_is_morning == True:
        prob_morning_dang = P_weekend_given_dang
        prob_morning_not_dang = P_weekend_given_not_dang
    else:
        prob_morning_dang = 1
        prob_morning_not_dang = 1

    PREDICT_CLASS = ''
    ans_dist_given_dang = p_x_given_y(front_end_district, dang_dist_mean, dang_dist_var)
    ans_month_given_dang = p_x_given_y(front_end_month, dang_month_mean, dang_month_var)

    ans_dist_not_given_dang = p_x_given_y(front_end_district, not_dang_dist_mean, not_dang_dist_var)
    ans_month_not_given_dang = p_x_given_y(front_end_month, not_dang_month_mean, not_dang_month_var)

    ans_prob_dang = P_dangerous*ans_dist_given_dang*ans_month_given_dang*prob_weekend_dang*prob_morning_dang

    ans_prob_not_dang = P_not_dangerous*ans_dist_not_given_dang*ans_month_not_given_dang*prob_morning_not_dang*prob_weekend_not_dang

    # print(ans_prob_dang, ans_prob_not_dang)
    diff = abs(ans_prob_not_dang - ans_prob_dang)
    if diff > 0.0004:
        PREDICT_CLASS = 'Not Dangerous'
    else:
        PREDICT_CLASS = 'Dangerous'

    return PREDICT_CLASS


def p_x_given_y(x, mean_y, variance_y):
    p = 1 / (math.sqrt(2 * math.pi * variance_y)) * math.exp((-(x - mean_y) ** 2) / (2 * variance_y))

    return p


def read_data():
    orig_df = pd.read_csv('Data_Ready_For_Naive_Bayes.csv')
    return orig_df


def main_algorithm(dist, weekend, month, morning):
    # data_df = prepare_and_clean_the_data()
    data_df = read_data()
    data_df['DIST_CODES'] = pd.factorize(data_df['DISTRICT'])[0] + 1

    district_code = data_df.loc[(data_df['DISTRICT'] == dist)]['DIST_CODES']
    district_code = district_code.tolist()[0]
    # weekend = True
    # month = 6
    # district_code = 3
    # morning = True

    if weekend == 'False':
        weekend = False
    else:
        weekend = True

    if morning == 'False':
        morning = False
    else:
        morning = True

    month = int(month)

    get_prediction = naive_bayes(data_df, weekend, month, district_code, morning)

    print('The class predicted is :', get_prediction)

    return get_prediction

if __name__ == "__main__":
    app.run(debug=True)
