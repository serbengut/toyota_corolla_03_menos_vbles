import pandas as pd

import streamlit as st
#from streamlit_echarts import st_echarts
from codigo_de_ejecucion_toyota_corolla_v02 import *
import numpy as np
import cloudpickle
import pickle

#Automcompletar rápido
#%config IPCompleter.greedy=True

from janitor import clean_names

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline



st.set_page_config(
     page_title = 'Toyota Corolla - Precio venta vehículo usado',
     #page_icon = 'DS4B_Logo_Blanco_Vertical_FB.png',
     layout = 'wide')

st.title('Toyota Corolla - Precio venta vehículo usado')


##### VBLE 1 #####
#s_n_abs_ = st.selectbox('ABS', ['Sí', 'No'])
#if s_n_abs_ == 'Sí':
#    abs_ = 1
#else:
#    abs_ = 0
abs_ = 1

##### VBLE 2 #####
#meses = st.slider('Meses', 1, 80)
meses=10

##### VBLE 3 #####
#s_n_airco = st.selectbox('A/A', ['Sí', 'No'])
#if s_n_airco == 'Sí':
#    airco = 1
#else:
#    airco = 0
airco = 1

##### VBLE 4 #####
s_n_automatic_airco = st.selectbox('Climatizador', ['Sí', 'No'])
if s_n_automatic_airco == 'Sí':
    automatic_airco = 1
else:
    automatic_airco = 0

##### VBLE 5 #####
cc = st.slider('cc', 1300, 2000)

##### VBLE 6 #####
#s_n_cd_player = st.selectbox('Reproductor de CD', ['Sí', 'No'])
#if s_n_cd_player == 'Sí':
#    cd_player = 1
#else:
#    cd_player = 0
cd_player = 1

##### VBLE 7 #####
color = st.selectbox('Color', ['Beige', 'Black' ,'Blue', 'Green', 'Grey', 'Red' ,'Silver', 'Violet' ,'White','Yellow'])


##### VBLE 8 #####
#gears = st.selectbox('Nº de marchas', [3, 4,6])
gears = 6

##### VBLE 9 #####
#guarantee_period = st.selectbox('Periodo de garantía', [3, 6,12,13,18,20,24,28,36])
guarantee_period = 24

##### VBLE 10 #####
#hp = st.slider('CV', 69, 192)
hp = 150

##### VBLE 11 #####
km = st.slider('km', 1, 243000)

##### VBLE 12 #####
#s_n_metallic_rim = st.selectbox('Metallic Rim', ['Sí', 'No'])
#if s_n_metallic_rim == 'Sí':
#    metallic_rim = 1
#else:
#    metallic_rim = 0
metallic_rim = 1

##### VBLE 13 #####   
#mfg_year = st.slider('MFR Year', 1998, 2004)
mfg_year=2000

##### VBLE 14 #####
#s_n_mfr_guarantee = st.selectbox('MFR Guarantee', ['Sí', 'No'])
#if s_n_mfr_guarantee == 'Sí':
#    mfr_guarantee = 1
#else:
#    mfr_guarantee = 0
mfr_guarantee = 1

##### VBLE 15 #####
#s_n_mistlamps = st.selectbox('Faros antiniebla', ['Sí', 'No'])
#if s_n_mistlamps == 'Sí':
#    mistlamps = 1
#else:
#    mistlamps = 0
mistlamps = 1

##### VBLE 16 #####
#model = st.selectbox('Modelo', ['TOYOTA Corolla 1.3 16V HATCHB LINEA TERRA 2/3-Doors', 'TOYOTA Corolla 1.3 16V LIFTB LINEA TERRA 4/5-Doors','TOYOTA Corolla 1.4 16V VVT I HATCHB TERRA 2/3-Doors','TOYOTA Corolla 1.6 16V HATCHB LINEA TERRA 2/3-Doors','TOYOTA Corolla 1.6 16V LIFTB LINEA LUNA 4/5-Doors','TOYOTA Corolla 1.6 16V LIFTB LINEA TERRA 4/5-Doors','TOYOTA Corolla 1.6 16V SEDAN LINEA TERRA 4/5-Doors','TOYOTA Corolla 1.6 16V VVT I LIFTB LUNA 4/5-Doors','TOYOTA Corolla 1.6 16V VVT I LIFTB SOL 4/5-Doors','TOYOTA Corolla 1.6 16V VVT I LIFTB TERRA 4/5-Doors','Otros'])
model = 'TOYOTA Corolla 1.3 16V HATCHB LINEA TERRA 2/3-Doors'

##### VBLE 17 #####
#s_n_powered_windows = st.selectbox('Elevalunas eléctricos', ['Sí', 'No'])
#if s_n_powered_windows == 'Sí':
#    powered_windows = 1
#else:
#    powered_windows = 0
powered_windows = 1

##### VBLE 18 #####
#quarterly_tax = st.slider('Quarterly Tax', 19, 283)
quarterly_tax = 150

##### VBLE 19 #####
#s_n_sport_model = st.selectbox('Modelo Deportivo', ['Sí', 'No'])
#if s_n_sport_model == 'Sí':
#    sport_model = 1
#else:
#    sport_model = 0
sport_model = 1

##### VBLE 20 #####
#weight = st.slider('Peso vehículo', 1000, 1615)
weight = 1600

registro = pd.DataFrame({'abs':abs_,
                         'age_08_04':meses,
                         'airco':airco,
                         'automatic_airco':automatic_airco,
                         'cc':cc,
                         'cd_player':cd_player,
                         'color':color,
                         'gears':gears,
                         'guarantee_period':guarantee_period,
                         'hp':hp,
                         'km':km,
                         'metallic_rim':metallic_rim,
                         'mfg_year':mfg_year,
                         'mfr_guarantee':mfr_guarantee,
                         'mistlamps':mistlamps,
                         'model':model,
                         'powered_windows':powered_windows,
                         'quarterly_tax':quarterly_tax,
                         'sport_model':sport_model,
                         'weight':weight}
                        ,index=[0])

registro

if st.sidebar.button('CALCULAR PRECIO'):

    precio = round(float(ejecutar_modelo(registro)),2)

    precio
else:
    st.write('DEFINE LOS PARÁMETROS DEL COCHE Y HAZ CLICK EN CALCULAR PRECIO')

