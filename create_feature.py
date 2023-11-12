# library

######################################################
# basic library
######################################################
from contextlib import contextmanager
from dataprep.datasets import get_dataset_names
from dataprep.eda import create_report
from glob import glob
from matplotlib_venn import venn2
from pathlib import Path
from pprint import pprint
from time import time

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import dtale
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import toml

# xfeat 量が多すぎなので1行記述を許可する
#from xfeat import SelectCategorical, LabelEncoder, Pipeline, ConcatCombination, SelectNumerical, ArithmeticCombinations, TargetEncoder, aggregation, GBDTFeatureSelector, GBDTFeatureExplorer
from xfeat import SelectCategorical, Pipeline, ConcatCombination, SelectNumerical, ArithmeticCombinations, TargetEncoder, aggregation, GBDTFeatureSelector, GBDTFeatureExplorer

import warnings
warnings.filterwarnings('ignore')

######################################################
# 環境設定 
######################################################
# OS環境設定
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Path定義設定 
PREFIX = Path().resolve().parent
PREFIX_INPUT_DATA = PREFIX / 'src/datasets/input'
PREFIX_OUTPUT_DATA = PREFIX / 'src/datasets/output'
PREFIX_TOML = PREFIX / 'src/def/'
print(f'PREFIX_INPUT_DATA: {PREFIX_INPUT_DATA}')
print(f'PREFIX_OUTPUT_DATA: {PREFIX_OUTPUT_DATA}')
print(f'PREFIX_TOML: {PREFIX_TOML}')

######################################################
# 独自ライブラリ取り込み向け Path定義
######################################################
MY_LIBRARY = str(f'{Path().resolve().parent}/src')
sys.path.append(MY_LIBRARY)

######################################################
# 独自ライブラリ import
######################################################
from features.common_features import convert_to_categorical_features
from features.common_features import create_stat_feature
from features.common_features import create_math_feature
from features.common_features import create_feature_abs_mean
from features.common_features import sturges_nbins
from features.common_features import create_feature_binning_labels
from features.common_features import create_feature_binning_labels_counter
from features.common_features import create_feature_clipped_only_upper
from features.common_features import create_feature_value_count_encoding
from features.common_features import create_feature_is_weekend
from features.common_features import create_feature_seasons
from features.common_features import create_feature_lead_time_to_standard_transform
from features.common_features import create_feature_adr_to_standard_transform
from features.common_features import create_feature_agent_to_standard_transform
from features.common_features import create_feature_relation_market_segment_and_deposite_type
from features.common_features import create_feature_relation_market_segment_and_customer_type
from features.common_features import create_feature_relation_distribution_channel_and_deposit_type
from features.common_features import create_feature_relation_deposit_type_and_customer_type
from features.common_features import create_feature_relation_market_segment_and_distribution_channel_and_deposit_type
from features.common_features import create_feature_relation_market_segment_and_distribution_channel_and_customer_type
from features.common_features import create_feature_relation_market_segment_and_deposit_type_and_customer_type
from features.common_features import create_feature_relation_distribution_channel_and_deposit_type_and_customer_type
from features.common_features import create_group_ave_reserved_days
from features.common_features import create_group_ave_stay_days
from features.common_features import create_group_ave_adr
from features.common_features import create_group_ave_cancel_rate

from features.timeseries_features import DateInfoTransform

###############################################################
# 業務ドメインfeatureライブラリ
###############################################################
from features.common_features import create_feature_family_size
from features.common_features import create_feature_all_stay_days
from features.common_features import create_feature_total_stay_costs
from features.common_features import create_feature_room_adr_per_nights
from features.common_features import create_feature_stay_adr_per_nights
from features.common_features import create_feature_travel_group
from features.common_features import create_feature_is_deposit
from features.common_features import create_feature_reservation_polularity
from features.common_features import create_feature_diff_roomtype_from_reserved_room_type_to_assigned_room_type
from features.common_features import create_feature_is_day_trip
from features.common_features import create_feature_high_cancel_rate_week
from features.common_features import create_feature_low_cancel_rate_week
#from features.common_features import create_feature_reservation_cancel_rate
from features.common_features import create_feature_previous_booking_minus_canceled_and_notcanceled
from features.common_features import create_feature_cancel_rate_low_high
from features.common_features import create_feature_cancel_singularity_classification

###############################################################
# gp生成featureライブラリ 
###############################################################
from features.gp_features import create_gp_feature_3133_X09
from features.gp_features import create_gp_feature_4900_X05
from features.gp_features import create_gp_feature_4900_X09
from features.gp_features import create_gp_feature_2693_X10
from features.gp_features import create_gp_feature_224_X08
from features.gp_features import create_gp_feature_731_X10
from features.gp_features import create_gp_feature_931_X09
from features.gp_features import create_gp_feature_4056_X07
from features.gp_features import create_gp_feature_118_X05
from features.gp_features import create_gp_feature_276_X05
from features.gp_features import create_gp_feature_4686_X07
from features.gp_features import create_gp_feature_4389_X03
from features.gp_features import create_gp_feature_800_X04
from features.gp_features import create_gp_feature_4861_X09
from features.gp_features import create_gp_feature_4083_X06
from features.gp_features import create_gp_feature_4083_X07
from features.gp_features import create_gp_feature_3168_X06
from features.gp_features import create_gp_feature_4625_X01
from features.gp_features import create_gp_feature_4981_X02
from features.gp_features import create_gp_feature_4981_X09
from features.gp_features import create_gp_feature_4981_X07
from features.gp_features import create_gp_feature_3471_X10
from features.gp_features import create_gp_feature_3643_X05
from features.gp_features import create_gp_feature_1540_X06
from features.gp_features import create_gp_feature_4900_X09
from features.gp_features import create_gp_feature_3992_X08
from features.gp_features import create_gp_feature_3137_X09
from features.gp_features import create_gp_feature_613_X07
from features.gp_features import create_gp_feature_2491_X04
from features.gp_features import create_gp_feature_2294_X10
from features.gp_features import create_gp_feature_4535_X03
from features.gp_features import create_gp_feature_3364_X02
from features.gp_features import create_gp_feature_1372_X04
from features.gp_features import create_gp_feature_59_X05
from features.gp_features import create_gp_feature_580_X09
from features.gp_features import create_gp_feature_4974_X08
from features.gp_features import create_gp_feature_3001_X02
from features.gp_features import create_gp_feature_3562_X05
from features.gp_features import create_gp_feature_792_X08
from features.gp_features import create_gp_feature_3099_X06
from features.gp_features import create_gp_feature_4111_X10
from features.gp_features import create_gp_feature_4409_X09
from features.gp_features import create_gp_feature_4831_X08
from features.gp_features import create_gp_feature_3883_X06
from features.gp_features import create_gp_feature_2349_X02
from features.gp_features import create_gp_feature_160_X08
from features.gp_features import create_gp_feature_1187_X03
from features.gp_features import create_gp_feature_931_X09
from features.gp_features import create_gp_feature_2179_X05
from features.gp_features import create_gp_feature_2179_X08
from features.gp_features import create_gp_feature_4021_X01
from features.gp_features import create_gp_feature_2574_X09
from features.gp_features import create_gp_feature_3007_X02
from features.gp_features import create_gp_feature_3923_X10
from features.gp_features import create_gp_feature_3664_X02
from features.gp_features import create_gp_feature_4663_X08
from features.gp_features import create_gp_feature_791_X07
from features.gp_features import create_gp_feature_2699_X04
from features.gp_features import create_gp_feature_3398_X10
from features.gp_features import create_gp_feature_372_X10
from features.gp_features import create_gp_feature_1903_X10
from features.gp_features import create_gp_feature_2699_X04
from features.gp_features import create_gp_feature_231_X04
from features.gp_features import create_gp_feature_1498_X07
from features.gp_features import create_gp_feature_915_X09
from features.gp_features import create_gp_feature_2146_X07
from features.gp_features import create_gp_feature_3527_X07
from features.gp_features import create_gp_feature_3684_X07
from features.gp_features import create_gp_feature_3796_X07
from features.gp_features import create_gp_feature_478_X10
from features.gp_features import create_gp_feature_4738_X02
from features.gp_features import create_gp_feature_3506_X10
from features.gp_features import create_gp_feature_2964_X06

###############################################################
# 定義toml読み込み
###############################################################
## feature関連
with open(f'{PREFIX_TOML}/features.toml') as f:
    DEF_FEATURES = toml.load(f)

###############################################################
# 固定値取り込み 
###############################################################
## 固定値取得
# lightGBM向けにカテゴリ変数列リストを定義する
## DataPrepの判定がベースだが見直しを要する
categorical_features = DEF_FEATURES['category']['categorical_features']
#pprint(categorical_features)

# キャンセル発生率取り込み
cancel_rate_train_average = DEF_FEATURES['cancel_rate']['cancel_rate_train_average']
#print(cancel_rate_train_average)

# StandardScaler変換対象カラム
ss_target_columns = DEF_FEATURES['StanderdScaler']['ss_target_columns']
#pprint(ss_target_columns)

# lightGBM best params
best_params_by_optuna = DEF_FEATURES['best_params_by_optuna']
#pprint(best_params_by_optuna)

# 数学変換適用対象カラム定義
math_feature_columns = DEF_FEATURES['math_feature_columns']['math_feature_columns']
#pprint(math_feature_columns)

# feature生成対象判定リスト定義 
select_feature_categories = DEF_FEATURES['select_feature_categories']['select_feature_categories']
#pprint(select_feature_categories)

###############################################################
# CSVデータ取り込み（コンペオリジナルから）
###############################################################
# data
train_df = pd.read_csv(f'{PREFIX_INPUT_DATA}/train.csv')
test_df = pd.read_csv(f'{PREFIX_INPUT_DATA}/test.csv')
sample_submission_df = pd.read_csv(f'{PREFIX_INPUT_DATA}/sample_submission.csv')

## train/test 縦結合
#  EDA処理コード簡易化のためリネーム
df = pd.concat([train_df, test_df], axis=0)

# データ件数概況
print(f"Length of train: {len(train_df)}")
print(f"Length of test:  {len(test_df)}")
print(f"Length of train+test:  {len(df)}")
print()

###############################################################
# Basicな欠損値対応 
###############################################################
# NA設定
df = df.fillna({
    'children': 0,     # children 2レコードがNA -> 0
    'agent': -1,       # agent 異例値扱い -> -1
    'country': '-1',   # country 異例扱い -> '-1'
})

# 型が想定外
df['children'] = df['children'].astype(int)

# companyは相手にしない
df = df.drop(columns=['company'], axis=1)

###############################################################
# Undefined対応 
###############################################################
if 'is_undefined' in select_feature_categories:
    # distribution_channel -> mode
    df['distribution_channel'] = np.where(df['distribution_channel']== "Undefined", "TA/TO", df['distribution_channel'])
    
    # meal ->  SC (no meal package)
    df['meal'] = np.where(df['meal']== "Undefined", "SC", df['meal'])
    
    # market_segment -> mode
    df['market_segment'] = np.where(df['market_segment']== "Undefined", "Online TA", df['market_segment'])

###############################################################
# adrマイナス問題／意味を考えるとありえない値
###############################################################
if 'is_minus_adr' in select_feature_categories:
    df['adr'] = abs(df['adr'])

###############################################################
# feature 生成 
###############################################################
# categorical feature定義
# 処理の都度で追加していく。。。
## initial setup
_categorical_features = [
    # 'ID',
    'hotel',
    # 'is_canceled',
    # 'lead_time',
    #'arrival_date_year',          # 検証対象から外す可能性あり
    #'arrival_date_month',         # 検証対象から外す可能性あり
    #'arrival_date_week_number',   # 検証対象から外す可能性あり
    #'arrival_date_day_of_month',  # 検証対象から外す可能性あり
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    #'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    #'previous_cancellations',          # 検証対象から外す可能性あり
    #'previous_bookings_not_canceled',  # 検証対象から外す可能性あり
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    # 'agent',
    #'days_in_waiting_list',
    'customer_type',
    # 'adr',
    #'required_car_parking_spaces',
    #'total_of_special_requests',
]

###############################################################
# datetime関連追加
#   yyyy-mm-dd     datetime64[ns]
#   yyyy-mm-dd_str object
#   yyyy-mm-week   object
#   yyyy-mm-week_UNIXTIME float
#   dev_UNIXTIME float
###############################################################
# datetime生成
dit = DateInfoTransform()
df = dit.arraival_date_to_datetime(df, format='%Y/%B/%d')
df['UNIXTIME'] = dit.arraival_date_to_UNIXTIME(df, format='%Y/%B/%d')

if 'is_datetime' in select_feature_categories:
    # 週末フラグ
    # 生成featureから_categorical_features listに追加
    df['is_weekend'] = create_feature_is_weekend(df)
    _categorical_features.append('is_weekend')
    
    # 季節フラグ
    # 生成featureから_categorical_features listに追加
    df['feature_seasons'] = create_feature_seasons(df)
    _categorical_features.append('feature_seasons')

###############################################################
# clipping 外れ値補正 -> 必ず実施したいところ
###############################################################
if 'is_clipping' in select_feature_categories:
    df.loc[df['lead_time']                      > 380, ['lead_time']                   ] = 380
    df.loc[df['stays_in_weekend_nights']        > 6, ['stays_in_weekend_nights']       ] = 6
    df.loc[df['stays_in_week_nights']           > 10, ['stays_in_week_nights']         ] = 10
    df.loc[df['adults']                         > 4, ['adults']                        ] = 4
    df.loc[df['children']                       > 8, ['children']                      ] = 0
    df.loc[df['babies']                         > 8, ['babies']                        ] = 0
    df.loc[df['days_in_waiting_list']           > 0, ['days_in_waiting_list']          ] = 1
    df.loc[df['previous_cancellations']         > 0, ['previous_cancellations']        ] = 1
    df.loc[df['previous_bookings_not_canceled'] > 0, ['previous_bookings_not_canceled']] = 1
    
# この処理は必須とする
# test_df['adr'].max()  # -> 5400 !!
# この日の全体レコードベースのadr平均で置き換える
query = (df['adr']==5400)
df.loc[query, 'adr'] = df[df['yyyy-mm-dd_str'] == '2016-03-25'].mean()['adr'].astype(int)    

###############################################################
# polynomial feature生成 / Numerical
###############################################################
if 'is_polynomial_numerical' in select_feature_categories:
    # 2次の多項式特徴量のクラスをインスタンス化
    pf = PolynomialFeatures(
        degree=3,                  # 多項式の次数
        #interaction_only=False,    # Trueの場合、ある特徴量を2乗以上した項が除かれる
        interaction_only=True,    # Trueの場合、ある特徴量を2乗以上した項が除かれる
        include_bias=True,         # Trueの場合、バイアス項を含める
        order='C'                  # 出力する配列の計算順序
        )
    
    num_target = list(SelectNumerical().fit_transform(df).columns)
    num_target = ['lead_time', 'adr', 'arrival_date_week_number','arrival_date_day_of_month']
    pf_columns = [
        'lead_time*adr',
        'lead_time*arrival_date_week_number',
        'lead_time*arrival_date_day_of_month',
        'adr*arrival_date_week_number',
        'adr*arrival_date_day_of_month',
        'arrival_date_week_number*arrival_date_day_of_month',
        'lead_time*adr*arrival_date_week_number',
        'lead_time*adr*arrival_date_day_of_month',
        'lead_time*arrival_date_week_number*arrival_date_day_of_month',
        'adr*arrival_date_week_number*arrival_date_day_of_month',
        ]
    _df = pd.DataFrame(pf.fit_transform(df[num_target]))
    _df = _df.reset_index()
    _df = _df.loc[:, 5:]
    _df.columns = pf_columns
    df = pd.concat([df.reset_index(drop=True), _df], axis=1)

    # 生成featureから_categorical_features listに追加

###############################################################
# polynomial feature生成 / Categrical
###############################################################
if 'is_polynomial_categorical' in select_feature_categories:
    encoder = Pipeline([
        # categorical feature選択
        SelectCategorical(exclude_cols=['yyyy-mm-dd', 'yyyy-mm-dd_str']),
        # `r=2` specifies the number of columns to combine the columns.
        ConcatCombination(
            drop_origin=True, 
            output_suffix="_cat_combination_2", 
            r=2),
    ])
    
    encoded_df = encoder.fit_transform(df)
    df = pd.concat([df, encoded_df], axis=1)

###############################################################
# math function agg
###############################################################
if 'is_math' in select_feature_categories:
    target_columns = ['yyyy-mm-week','lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','adr']
    df = create_math_feature(df, target_key='yyyy-mm-week', target_columns=target_columns)

###############################################################
# mean distance 
###############################################################
if 'is_mean_distance' in select_feature_categories:
    df['mean_distance_lead_time'], \
    df['mean_distance_stays_in_weekend_nights'], \
    df['mean_distance_stays_in_week_nights'], \
    df['mean_distance_adutls'], \
    df['mean_distance_adr'] = create_feature_abs_mean(df)

###############################################################
# binning 
###############################################################
if 'is_binning' in select_feature_categories:
    df['binning_labels_lead_time'] = create_feature_binning_labels(df, 'lead_time', bins=sturges_nbins(df['lead_time']))
    df['binning_labels_adr'] = create_feature_binning_labels(df, 'adr', bins=sturges_nbins(df['adr']))
    df = create_feature_binning_labels_counter(df, 'lead_time', bins=sturges_nbins(df['lead_time']))
    df = create_feature_binning_labels_counter(df, 'adr', bins=sturges_nbins(df['adr']))
    # 個別対応->all 0 featureになっているとpycaret:stackingでエラーとなる
    df = df.drop(columns=['binning_labels_adr_468_494'], axis=1)
    
###############################################################
# value counts 
###############################################################
if 'is_value_counts' in select_feature_categories:
    target_columns = [
        'lead_time',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'adults',
        'market_segment',
        'distribution_channel',
        'reserved_room_type',
        'assigned_room_type',
        'deposit_type',
        'agent',
        'customer_type',
        'adr',
    ]
    df = create_feature_value_count_encoding(df, target_columns)    
    
###############################################################
# 分散標準化 
###############################################################
if 'is_standard' in select_feature_categories:
    df['starndard_lead_time'] = create_feature_lead_time_to_standard_transform(df)
    df['starndard_adr'] = create_feature_adr_to_standard_transform(df)
    df['starndard_agent'] = create_feature_agent_to_standard_transform(df)

###############################################################
# グルーピング relation系 
###############################################################
if 'is_grouping' in select_feature_categories:
    df['relation_market_segment_and_deposite_type'] = create_feature_relation_market_segment_and_deposite_type(df)
    df['relation_market_segment_and_customer_type'] = create_feature_relation_market_segment_and_customer_type(df)
    df['relation_distribution_channel_and_deposit_type'] = create_feature_relation_distribution_channel_and_deposit_type(df)
    df['relation_deposit_type_and_customer_type'] = create_feature_relation_deposit_type_and_customer_type(df)
    df['relation_market_segment_and_distribution_channel_and_deposit_type'] = create_feature_relation_market_segment_and_distribution_channel_and_deposit_type(df)
    df['relation_market_segment_and_distribution_channel_and_customer_type'] = create_feature_relation_market_segment_and_distribution_channel_and_customer_type(df)
    df['relation_market_segment_and_deposit_type_and_customer_type'] = create_feature_relation_market_segment_and_deposit_type_and_customer_type(df)
    df['relation_distribution_channel_and_deposit_type_and_customer_type'] = create_feature_relation_distribution_channel_and_deposit_type_and_customer_type(df)    
    
###############################################################
# 平均予約数、平均滞在日数、平均価格、平均キャンセル率 
# ただしis_canceledにたいして高すぎる相関は危険なfeatureになっている可能性がある
###############################################################
if 'is_averaging' in select_feature_categories:
    # シナリオ 
    # 強すぎる相関を外す > 0.40 f1_scoreが0.91ってありえない結果
    # 個別に外す 1 f1_score: 0.8595665477269193
    #   - ['customer_type', 'adr', 'required_car_parking_spaces', 'arrival_date_year', 'arrival_date_day_of_month']
    
    # 個別に外す 2 f1_score: 0.8592193636478083
    #   - ['customer_type', 'adr', 'required_car_parking_spaces', 'arrival_date_year', 'arrival_date_day_of_month']
    #   - ['arrival_date_month', 'market_segment', 'total_of_special_requests', 'arrival_date_day_of_month', 'required_car_parking_spaces', 'arrival_date_week_number'] # 0.39
    
    # 個別に外す 3 f1_score: 0.8451629311466394
    #   - ['customer_type', 'adr', 'required_car_parking_spaces', 'arrival_date_year', 'arrival_date_day_of_month']
    #   - ['arrival_date_month', 'market_segment', 'total_of_special_requests', 'arrival_date_day_of_month', 'required_car_parking_spaces', 'arrival_date_week_number'] # 0.39
    #   - ['country', 'arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'is_repeated_guest', 'customer_type']
    #   - ['country', 'deposit_type']
    
    # 個別に外す 4 f1_score: 0.8464057297305331
    #   - ['customer_type', 'adr', 'required_car_parking_spaces', 'arrival_date_year', 'arrival_date_day_of_month']
    #   - ['arrival_date_month', 'market_segment', 'total_of_special_requests', 'arrival_date_day_of_month', 'required_car_parking_spaces', 'arrival_date_week_number'] # 0.39
    #   - ['country', 'arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'is_repeated_guest', 'customer_type']
    #   - ['country', 'deposit_type']
    #   - ['arrival_date_month', 'agent', 'arrival_date_week_number', 'total_of_special_requests']
    #   - ['arrival_date_month', 'agent', 'stays_in_weekend_nights', 'arrival_date_week_number', 'previous_cancellations']
    
    # 個別に外す 5 f1_score: 0.8333357501326096 
    #   - ['customer_type', 'adr', 'required_car_parking_spaces', 'arrival_date_year', 'arrival_date_day_of_month']
    #   - ['arrival_date_month', 'market_segment', 'total_of_special_requests', 'arrival_date_day_of_month', 'required_car_parking_spaces', 'arrival_date_week_number'] # 0.39
    #   - ['country', 'arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'is_repeated_guest', 'customer_type']
    #   - ['country', 'deposit_type']
    #   - ['arrival_date_month', 'agent', 'arrival_date_week_number', 'total_of_special_requests']
    #   - ['arrival_date_month', 'agent', 'stays_in_weekend_nights', 'arrival_date_week_number', 'previous_cancellations']
    #   - ['arrival_date_month', 'agent', 'arrival_date_week_number', 'total_of_special_requests']
    #   - ['previous_bookings_not_canceled', 'total_of_special_requests', 'arrival_date_month']
    
    
    ####################################################
    ## 平均予約数（平均キャンセル数）
    ####################################################
    #
    ## 0.55 corrは高いがSHAPとしては影響が少ない謎feature
    ##query = ['market_segment', 'deposit_type', 'previous_cancellations']
    ##df = create_group_ave_reserved_days(df, query)
    #
    ## corr: 0.43 corrは高いがSHAPとしては影響が少ない謎feature
    ##query = ['country', 'children', 'arrival_date_day_of_month']
    ##df = create_group_ave_reserved_days(df, query)
    #
    ## 0.39
    ## マイナス寄与SHAP高い
    ##query = ['arrival_date_month', 'market_segment', 'total_of_special_requests', 'arrival_date_day_of_month', 'required_car_parking_spaces', 'arrival_date_week_number'] # 0.39
    ##df = create_group_ave_reserved_days(df, query)
    #
    ## マイナス寄与高い SHAP 確実に過学習
    ## 0.38
    ##query = ['customer_type', 'adr', 'required_car_parking_spaces', 'arrival_date_year', 'arrival_date_day_of_month']
    ##df = create_group_ave_reserved_days(df, query)
    #
    # マイナス寄与高い SHAP
    ## 0.34
    #query = ['country', 'arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'is_repeated_guest', 'customer_type']
    #df = create_group_ave_reserved_days(df, query)
    
    # 0.34
    query = ['country', 'deposit_type']
    df = create_group_ave_reserved_days(df, query)
    
    # マイナス寄与高い SHAP
    ## 0.30
    #query = ['arrival_date_month', 'agent', 'arrival_date_week_number', 'total_of_special_requests']
    #df = create_group_ave_reserved_days(df, query)
    
    # SHAP寄与高め
    # 0.29
    query = ['previous_bookings_not_canceled', 'total_of_special_requests', 'arrival_date_month']
    df = create_group_ave_reserved_days(df, query)
    
    # マイナス寄与高い SHAP
    ## 0.24
    #query = ['arrival_date_month', 'agent', 'stays_in_weekend_nights', 'arrival_date_week_number', 'previous_cancellations']
    #df = create_group_ave_reserved_days(df, query)
    
    # 0.19
    query = ['stays_in_weekend_nights', 'stays_in_week_nights', 'market_segment', 'distribution_channel']
    df = create_group_ave_reserved_days(df, query)
    
    # 0.14
    query = ['arrival_date_month', 'is_repeated_guest', 'arrival_date_week_number']
    df = create_group_ave_reserved_days(df, query)
    
    # 0.14
    query = ['stays_in_weekend_nights', 'stays_in_week_nights', 'customer_type']
    df = create_group_ave_reserved_days(df, query)
    
    # SHAPマイナス寄与高め
    # 0.12
    #query = ['agent', 'deposit_type', 'arrival_date_month']
    #df = create_group_ave_reserved_days(df, query)
    
    # 0.10
    query = ['customer_type', 'agent', 'days_in_waiting_list']
    df = create_group_ave_reserved_days(df, query)
    
    
    ####################################################
    # 平均滞在日数 -> あってもなくてもほぼ関係ない
    ####################################################
    #query = ['arrival_date_month', 'is_repeated_guest', 'arrival_date_week_number'] # 0.14
    #df = create_group_ave_stay_days(df, query)
    
    ####################################################
    ## 平均価格 ボツにする予定
    ####################################################
    #query = ['arrival_date_week_number', 'deposit_type', 'agent', 'assigned_room_type'] # 0.05
    #df = create_group_ave_adr(df, query)
    #
    ####################################################
    ## キャンセル率
    ####################################################
    # 0.57 ほとんど影響なし
    #query = ['deposit_type', 'agent', 'assigned_room_type']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.56  f1_score: 0.8435797593375398 -> 単独
    #query = ['arrival_date_week_number', 'arrival_date_year', 'market_segment', 'arrival_date_day_of_month', 'meal']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.55 f1_score: 0.8273050617273265 -> 単独　悪化。。。SHAP的にはプラス相関には見えるのだが
    #query = ['customer_type', 'total_of_special_requests', 'agent']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.54 SHAPでは±寄与大きめのfeature
    #query = ['arrival_date_week_number', 'customer_type', 'is_repeated_guest', 'deposit_type', 'total_of_special_requests']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.50 SHAPでは寄与＋大きめのfeature だけどlightGBMは悪化
    #query = ['arrival_date_month', 'customer_type', 'previous_bookings_not_canceled', 'deposit_type']
    #df = create_group_ave_cancel_rate(df, query)
    #
    # 0.44 SHAPでは寄与大きいがマイナス寄与率が高め
    #query = ['previous_bookings_not_canceled', 'assigned_room_type', 'arrival_date_week_number', 'arrival_date_day_of_month', 'required_car_parking_spaces']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.40 あんまりパットしないSHAPだがSTDは高い
    query = ['agent', 'days_in_waiting_list']
    df = create_group_ave_cancel_rate(df, query)
    
    #0.40 あんまりパットしないSHAPだがSTDは高い
    query = ['previous_cancellations', 'total_of_special_requests', 'arrival_date_month', 'is_repeated_guest']
    df = create_group_ave_cancel_rate(df, query)
    
    # 0.35
    query = ['arrival_date_month','total_of_special_requests', 'customer_type']
    df = create_group_ave_cancel_rate(df, query)
    #
    # 0.34
    query = ['customer_type', 'total_of_special_requests', 'previous_bookings_not_canceled']
    df = create_group_ave_cancel_rate(df, query)
    
    # 0.29
    query = ['arrival_date_month', 'is_repeated_guest', 'market_segment']
    df = create_group_ave_cancel_rate(df, query)
    
    # 0.22 SHAPマイナス寄与高い
    #query = ['assigned_room_type', 'is_repeated_guest', 'arrival_date_month', 'arrival_date_week_number']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.22 SHAPマイナス寄与高い
    #query = ['arrival_date_month', 'is_repeated_guest', 'required_car_parking_spaces']
    #df = create_group_ave_cancel_rate(df, query)
    
    # 0.15
    query = ['arrival_date_year', 'children', 'arrival_date_week_number']
    df = create_group_ave_cancel_rate(df, query)
    
    # 0.14
    query = ['is_repeated_guest', 'previous_bookings_not_canceled', 'reserved_room_type'] # 0.14
    df = create_group_ave_cancel_rate(df, query)
    
    # 0.12
    query = ['adults', 'children', 'babies']
    df = create_group_ave_cancel_rate(df, query)


###############################################################
# domain 人数算出
###############################################################
if 'is_family_size' in select_feature_categories:
    df['domain_family_size'] = create_feature_family_size(df)

###############################################################
# domain 総宿泊数 
###############################################################
if 'is_all_stay_days' in select_feature_categories:
    df['domain_all_stay_days'] = create_feature_all_stay_days(df)

###############################################################
# domain 宿泊一泊あたりの人単価平均 
###############################################################
if 'is_adr_room_per_nights' in select_feature_categories:
    df['domain_room_adr_per_nights'] = create_feature_room_adr_per_nights(df)
    
###############################################################
# domain 宿泊一連あたりの人単価平均 
###############################################################
if 'is_adr_stay_per_nights' in select_feature_categories:
    df['doamin_stay_adr_per_nights'] = create_feature_stay_adr_per_nights(df)

###############################################################
# domain 宿泊タイプ分類 
###############################################################
if 'is_travel_group' in select_feature_categories:
    df['domain_travel_group'] = df.apply(
        lambda x: create_feature_travel_group(x['adults'], x['children'], x['babies']),
        axis=1,
        )

###############################################################
# domain deposit判定 
###############################################################
if 'is_deposit' in select_feature_categories:
    df['domain_is_deposit'] = df.apply(
        lambda x: create_feature_is_deposit(x['deposit_type']),
        axis=1,
        )

###############################################################
# domain popularity 
###############################################################
if 'is_popularity' in select_feature_categories:
    df['domain_reservation_polularity'] = create_feature_reservation_polularity(df)

###############################################################
# domain diff_room_type
###############################################################
if 'is_diff_room_type' in select_feature_categories:
    df['domain_diff_roomtype_from_reserved_room_type_to_assigned_room_type'] = create_feature_diff_roomtype_from_reserved_room_type_to_assigned_room_type(df)

###############################################################
# domain 日帰り判定
###############################################################
if 'is_day_trip' in select_feature_categories:
    df['domain_is_day_trip'] = create_feature_is_day_trip(df)

###############################################################
# domain キャンセル率 
###############################################################
if 'is_cancel_rate' in select_feature_categories:
    df['domain_high_cancel_rate_week'] = create_feature_high_cancel_rate_week(df)
    df['domain_low_cancel_rate_week'] = create_feature_low_cancel_rate_week(df)
    df['domain_previous_booking_minus_canceled_and_notcanceled'] = create_feature_previous_booking_minus_canceled_and_notcanceled(df)

###############################################################
# domain 二項分布ベースでの日別キャンセル特異点 
###############################################################
if 'is_binomial' in select_feature_categories:
    df = create_feature_cancel_rate_low_high(df, cancel_rate=cancel_rate_train_average)
    
###############################################################
# domain キャンセル罰則境界 
###############################################################
if 'is_cancel_singularity' in select_feature_categories:
    df['dev_cancel_singularity_classification'] = df['lead_time'].map(create_feature_cancel_singularity_classification)

print(df.shape)



###############################################################
# categorical変換 
###############################################################
# arraival_date_monthは個別にマッピング
## 月名をアルファベット順にカテゴリエンコーダしてくるとは。。。
arraival_date_month_mapping = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12,
}
df['arrival_date_month'] = df['arrival_date_month'].map(arraival_date_month_mapping)

room_type_mapping = {
    'A':  1,
    'B':  2,
    'C':  3,
    'D':  4,
    'E':  5,
    'F':  6,
    'G':  7,
    'H':  8,
    'I':  9,
    'J':  11,
    'K':  12,
    'L':  13,
    'M':  14,
    'N':  15,
    'O':  16,
    'P':  17,
    }

# 先に個別対応 Label encoder
df['reserved_room_type'] = df['reserved_room_type'].map(room_type_mapping)
df['assigned_room_type'] = df['assigned_room_type'].map(room_type_mapping)

# label encoder
le = LabelEncoder()

# カテゴリカル変換候補
_categorical_features = list(SelectCategorical().fit_transform(df).columns)
#_categorical_features.remove('ID')
#_categorical_features.remove('is_canceled')
_categorical_features.remove('yyyy-mm-dd')
_categorical_features.remove('yyyy-mm-dd_str')
_categorical_features.remove('yyyy-mm-week')
#_categorical_features.remove('arrival_date_month')
#_categorical_features.remove('p_low')
#_categorical_features.remove('p_high')
_categorical_features.remove('binning_labels_lead_time'),
_categorical_features.remove('binning_labels_adr'),

print(_categorical_features)

# 変換対象外を指定
for column in _categorical_features:
    # カテゴリカル属性定義（除くarrival_date_month)に対してカテゴリカル変換実施
    df[column] = le.fit_transform(df[column])
    print(column, le.classes_)

###############################################################
# target encoding & train/test分割
###############################################################
if 'is_target_encoding' in select_feature_categories:
    # categorical target encoder
    #_categorical_features = list(SelectCategorical().fit_transform(df).columns)
    print('='*80)
    pprint(_categorical_features)
    print('='*80)
    
    ## target encoder
    fold = KFold(n_splits=5, shuffle=False)
    encoder = TargetEncoder(
        input_cols=_categorical_features,  # 変換前のCategorical対象を利用する
        target_col=['is_canceled'],
        fold=fold,
        output_suffix='_target_encoder',
    )





    ## ここから少しややこしい
    ## trainでtaget encoder学習 -> train学習結果で全体学習(df) -> 全体学習結果からtest_dfを切り出す
    # tranでtarget encoder
    train_df = df[:len(train_df)]
    train_df = encoder.fit_transform(train_df) 

    # train学習結果で全体処理
    df = encoder.transform(df)

    # 全体学習結果からtest_dfを生成・分離
    # NAN前提でくっついているのでtest_dfからis_canceledを剥がす
    test_df = df[len(train_df):]
    test_df = test_df.drop(columns=['is_canceled'], axis=0)    

else:
    # target encoderを行わずに単純分離 df -> train_df, test_df
    train_df = df[:len(train_df)]
    test_df = df[len(train_df):]
    test_df = test_df.drop(columns=['is_canceled'], axis=0)    

###############################################################
# gp feature作成 
###############################################################
if 'is_gp' in select_feature_categories:
    train_df['gp_feature_3133_X09'] = create_gp_feature_3133_X09(train_df)
    train_df['gp_feature_4900_X05'] = create_gp_feature_4900_X05(train_df)
    train_df['gp_feature_4900_X09'] = create_gp_feature_4900_X09(train_df)
    train_df['gp_feature_2693_X10'] = create_gp_feature_2693_X10(train_df)
    train_df['gp_feature_224_X08']  = create_gp_feature_224_X08(train_df)
    train_df['gp_feature_731_X10']  = create_gp_feature_731_X10(train_df)
    train_df['gp_feature_931_X09']  = create_gp_feature_931_X09(train_df)
    train_df['gp_feature_4056_X07'] = create_gp_feature_4056_X07(train_df)
    train_df['gp_feature_118_X05']  = create_gp_feature_118_X05(train_df)
    train_df['gp_feature_276_X05']  = create_gp_feature_276_X05(train_df)
    train_df['gp_feature_4686_X07'] = create_gp_feature_4686_X07(train_df)
    train_df['gp_feature_4389_X03'] = create_gp_feature_4389_X03(train_df)
    train_df['gp_feature_800_X04']  = create_gp_feature_800_X04(train_df)
    train_df['gp_feature_4861_X09'] = create_gp_feature_4861_X09(train_df)
    train_df['gp_feature_4083_X06'] = create_gp_feature_4083_X06(train_df)
    train_df['gp_feature_4083_X07'] = create_gp_feature_4083_X07(train_df)
    train_df['gp_feature_3168_X06'] = create_gp_feature_3168_X06(train_df)
    train_df['gp_feature_4625_X01'] = create_gp_feature_4625_X01(train_df)
    train_df['gp_feature_4981_X02'] = create_gp_feature_4981_X02(train_df)
    train_df['gp_feature_4981_X09'] = create_gp_feature_4981_X09(train_df)
    train_df['gp_feature_4981_X07'] = create_gp_feature_4981_X07(train_df)
    train_df['gp_feature_3471_X10'] = create_gp_feature_3471_X10(train_df)
    train_df['gp_feature_3643_X05'] = create_gp_feature_3643_X05(train_df)
    train_df['gp_feature_1540_X06'] = create_gp_feature_1540_X06(train_df)
    train_df['gp_feature_2349_X02'] = create_gp_feature_2349_X02(train_df)
    train_df['gp_feature_3992_X08'] = create_gp_feature_3992_X08(train_df)
    train_df['gp_feature_3137_X09'] = create_gp_feature_3137_X09(train_df)
    train_df['gp_feature_613_X07']  = create_gp_feature_613_X07(train_df)
    train_df['gp_feature_2491_X04'] = create_gp_feature_2491_X04(train_df)
    train_df['gp_feature_2294_X10'] = create_gp_feature_2294_X10(train_df)
    train_df['gp_feature_4535_X03'] = create_gp_feature_4535_X03(train_df)
    train_df['gp_feature_3364_X02'] = create_gp_feature_3364_X02(train_df)
    train_df['gp_feature_1372_X04'] = create_gp_feature_1372_X04(train_df)
    train_df['gp_feature_59_X05']   = create_gp_feature_59_X05(train_df)
    train_df['gp_feature_580_X09']  = create_gp_feature_580_X09(train_df)
    train_df['gp_feature_4974_X08'] = create_gp_feature_4974_X08(train_df)
    train_df['gp_feature_3001_X02'] = create_gp_feature_3001_X02(train_df)
    train_df['gp_feature_3562_X05'] = create_gp_feature_3562_X05(train_df)
    train_df['gp_feature_792_X08']  = create_gp_feature_792_X08(train_df)
    train_df['gp_feature_3099_X06'] = create_gp_feature_3099_X06(train_df)
    train_df['gp_feature_4111_X10'] = create_gp_feature_4111_X10(train_df)
    train_df['gp_feature_4409_X09'] = create_gp_feature_4409_X09(train_df)
    train_df['gp_feature_4831_X08'] = create_gp_feature_4831_X08(train_df)
    train_df['gp_feature_3883_X06'] = create_gp_feature_3883_X06(train_df)
    train_df['gp_feature_2349_X02'] = create_gp_feature_2349_X02(train_df)
    train_df['gp_feature_160_X08']  = create_gp_feature_160_X08(train_df)
    train_df['gp_feature_1187_X03'] = create_gp_feature_1187_X03(train_df)
    train_df['gp_feature_931_X09']  = create_gp_feature_931_X09(train_df)
    train_df['gp_feature_2179_X05'] = create_gp_feature_2179_X05(train_df)
    train_df['gp_feature_2179_X08'] = create_gp_feature_2179_X08(train_df)
    train_df['gp_feature_4021_X01'] = create_gp_feature_4021_X01(train_df)
    train_df['gp_feature_2574_X09'] = create_gp_feature_2574_X09(train_df)
    train_df['gp_feature_3007_X02'] = create_gp_feature_3007_X02(train_df)
    train_df['gp_feature_3923_X10'] = create_gp_feature_3923_X10(train_df)
    train_df['gp_feature_3664_X02'] = create_gp_feature_3664_X02(train_df)
    train_df['gp_feature_4663_X08'] = create_gp_feature_4663_X08(train_df)
    train_df['gp_feature_791_X07']  = create_gp_feature_791_X07(train_df)
    train_df['gp_feature_2699_X04'] = create_gp_feature_2699_X04(train_df)
    train_df['gp_feature_3398_X10'] = create_gp_feature_3398_X10(train_df)
    train_df['gp_feature_372_X10']  = create_gp_feature_372_X10(train_df)
    train_df['gp_feature_1903_X10'] = create_gp_feature_1903_X10(train_df)
    train_df['gp_feature_2699_X04'] = create_gp_feature_2699_X04(train_df)
    train_df['gp_feature_231_X04']  = create_gp_feature_231_X04(train_df)
    train_df['gp_feature_1498_X07'] = create_gp_feature_1498_X07(train_df)
    train_df['gp_feature_915_X09']  = create_gp_feature_915_X09(train_df)
    train_df['gp_feature_2146_X07'] = create_gp_feature_2146_X07(train_df)
    train_df['gp_feature_3527_X07'] = create_gp_feature_3527_X07(train_df)
    train_df['gp_feature_3684_X07'] = create_gp_feature_3684_X07(train_df)
    train_df['gp_feature_3796_X07'] = create_gp_feature_3796_X07(train_df)
    train_df['gp_feature_478_X10']  = create_gp_feature_478_X10(train_df)
    train_df['gp_feature_4738_X02'] = create_gp_feature_4738_X02(train_df)
    train_df['gp_feature_3506_X10'] = create_gp_feature_3506_X10(train_df)
    train_df['gp_feature_2964_X06'] = create_gp_feature_2964_X06(train_df)

    test_df['gp_feature_3133_X09'] = create_gp_feature_3133_X09(test_df)
    test_df['gp_feature_4900_X05'] = create_gp_feature_4900_X05(test_df)
    test_df['gp_feature_4900_X09'] = create_gp_feature_4900_X09(test_df)
    test_df['gp_feature_2693_X10'] = create_gp_feature_2693_X10(test_df)
    test_df['gp_feature_224_X08']  = create_gp_feature_224_X08(test_df)
    test_df['gp_feature_731_X10']  = create_gp_feature_731_X10(test_df)
    test_df['gp_feature_931_X09']  = create_gp_feature_931_X09(test_df)
    test_df['gp_feature_4056_X07'] = create_gp_feature_4056_X07(test_df)
    test_df['gp_feature_118_X05']  = create_gp_feature_118_X05(test_df)
    test_df['gp_feature_276_X05']  = create_gp_feature_276_X05(test_df)
    test_df['gp_feature_4686_X07'] = create_gp_feature_4686_X07(test_df)
    test_df['gp_feature_4389_X03'] = create_gp_feature_4389_X03(test_df)
    test_df['gp_feature_800_X04']  = create_gp_feature_800_X04(test_df)
    test_df['gp_feature_4861_X09'] = create_gp_feature_4861_X09(test_df)
    test_df['gp_feature_4083_X06'] = create_gp_feature_4083_X06(test_df)
    test_df['gp_feature_4083_X07'] = create_gp_feature_4083_X07(test_df)
    test_df['gp_feature_3168_X06'] = create_gp_feature_3168_X06(test_df)
    test_df['gp_feature_4625_X01'] = create_gp_feature_4625_X01(test_df)
    test_df['gp_feature_4981_X02'] = create_gp_feature_4981_X02(test_df)
    test_df['gp_feature_4981_X09'] = create_gp_feature_4981_X09(test_df)
    test_df['gp_feature_4981_X07'] = create_gp_feature_4981_X07(test_df)
    test_df['gp_feature_3471_X10'] = create_gp_feature_3471_X10(test_df)
    test_df['gp_feature_3643_X05'] = create_gp_feature_3643_X05(test_df)
    test_df['gp_feature_1540_X06'] = create_gp_feature_1540_X06(test_df)
    test_df['gp_feature_2349_X02'] = create_gp_feature_4900_X09(test_df)
    test_df['gp_feature_3992_X08'] = create_gp_feature_3992_X08(test_df)
    test_df['gp_feature_3137_X09'] = create_gp_feature_3137_X09(test_df)
    test_df['gp_feature_613_X07']  = create_gp_feature_613_X07(test_df)
    test_df['gp_feature_2491_X04'] = create_gp_feature_2491_X04(test_df)
    test_df['gp_feature_2294_X10'] = create_gp_feature_2294_X10(test_df)
    test_df['gp_feature_4535_X03'] = create_gp_feature_4535_X03(test_df)
    test_df['gp_feature_3364_X02'] = create_gp_feature_3364_X02(test_df)
    test_df['gp_feature_1372_X04'] = create_gp_feature_1372_X04(test_df)
    test_df['gp_feature_59_X05']   = create_gp_feature_59_X05(test_df)
    test_df['gp_feature_580_X09']  = create_gp_feature_580_X09(test_df)
    test_df['gp_feature_4974_X08'] = create_gp_feature_4974_X08(test_df)
    test_df['gp_feature_3001_X02'] = create_gp_feature_3001_X02(test_df)
    test_df['gp_feature_3562_X05'] = create_gp_feature_3562_X05(test_df)
    test_df['gp_feature_792_X08']  = create_gp_feature_792_X08(test_df)
    test_df['gp_feature_3099_X06'] = create_gp_feature_3099_X06(test_df)
    test_df['gp_feature_4111_X10'] = create_gp_feature_4111_X10(test_df)
    test_df['gp_feature_4409_X09'] = create_gp_feature_4409_X09(test_df)
    test_df['gp_feature_4831_X08'] = create_gp_feature_4831_X08(test_df)
    test_df['gp_feature_3883_X06'] = create_gp_feature_3883_X06(test_df)
    test_df['gp_feature_2349_X02'] = create_gp_feature_2349_X02(test_df)
    test_df['gp_feature_160_X08']  = create_gp_feature_160_X08(test_df)
    test_df['gp_feature_1187_X03'] = create_gp_feature_1187_X03(test_df)
    test_df['gp_feature_931_X09']  = create_gp_feature_931_X09(test_df)
    test_df['gp_feature_2179_X05'] = create_gp_feature_2179_X05(test_df)
    test_df['gp_feature_2179_X08'] = create_gp_feature_2179_X08(test_df)
    test_df['gp_feature_4021_X01'] = create_gp_feature_4021_X01(test_df)
    test_df['gp_feature_2574_X09'] = create_gp_feature_2574_X09(test_df)
    test_df['gp_feature_3007_X02'] = create_gp_feature_3007_X02(test_df)
    test_df['gp_feature_3923_X10'] = create_gp_feature_3923_X10(test_df)
    test_df['gp_feature_3664_X02'] = create_gp_feature_3664_X02(test_df)
    test_df['gp_feature_4663_X08'] = create_gp_feature_4663_X08(test_df)
    test_df['gp_feature_791_X07']  = create_gp_feature_791_X07(test_df)
    test_df['gp_feature_2699_X04'] = create_gp_feature_2699_X04(test_df)
    test_df['gp_feature_3398_X10'] = create_gp_feature_3398_X10(test_df)
    test_df['gp_feature_372_X10']  = create_gp_feature_372_X10(test_df)
    test_df['gp_feature_1903_X10'] = create_gp_feature_1903_X10(test_df)
    test_df['gp_feature_2699_X04'] = create_gp_feature_2699_X04(test_df)
    test_df['gp_feature_231_X04']  = create_gp_feature_231_X04(test_df)
    test_df['gp_feature_1498_X07'] = create_gp_feature_1498_X07(test_df)
    test_df['gp_feature_915_X09']  = create_gp_feature_915_X09(test_df)
    test_df['gp_feature_2146_X07'] = create_gp_feature_2146_X07(test_df)
    test_df['gp_feature_3527_X07'] = create_gp_feature_3527_X07(test_df)
    test_df['gp_feature_3684_X07'] = create_gp_feature_3684_X07(test_df)
    test_df['gp_feature_3796_X07'] = create_gp_feature_3796_X07(test_df)
    test_df['gp_feature_478_X10']  = create_gp_feature_478_X10(test_df)
    test_df['gp_feature_4738_X02'] = create_gp_feature_4738_X02(test_df)
    test_df['gp_feature_3506_X10'] = create_gp_feature_3506_X10(test_df)
    test_df['gp_feature_2964_X06'] = create_gp_feature_2964_X06(test_df)

###############################################################
# 強制置き換え train_df
# データに何らかの不備があると仮定して対処
###############################################################
if 'is_error_replace' in select_feature_categories:
    # 宿泊人数すべて0なのにis_canceled==0 ってどういうこと
    # ただし打ち間違いの可能性もある
    ## 条件つけてadr==0の場合に限り、is_canceled -> 1 に書き換える
    ## - adult/children/babiesがすべてゼロ
    ## - adrがゼロ
    ## - is_canceledが0となっているレコード
    query = (train_df['adults'] == 0) & (train_df['children'] == 0) & (train_df['babies'] == 0) & (train_df['adr'] == 0) & (train_df['is_canceled'] == 0)
    train_df.loc[query, 'is_canceled'] = 1
    
    # agentがマイナスのときに is_canceled -> 1に書き換える
    # 怪しい予約フラグ
    # - agentが0以下
    # - previous_bookings_not_canceled - previous_cancellations > 5
    # - is_canceledが0となっているレコード
    query = (train_df['agent'] <= 0) & ((train_df['previous_bookings_not_canceled'] - train_df['previous_cancellations']) > 5)
    train_df.loc[query, 'is_canceled'] = 1
    
    # adrがマイナスのときに is_canceled -> 1に書き換える
    # - adr がマイナス
    # - country == 135 怪しすぎる
    query = (train_df['adr'] <= 0) & (train_df['country'] == 135 )
    train_df.loc[query, 'is_canceled'] = 1

    # dev_reservation_polularityが0のときに is_canceled -> 1に書き換える
    # - cancel及びnotcancel合算が０
    # - adults が０
    # - この予約成立はあやしいでしょ
    query = ((train_df['previous_bookings_not_canceled'] + train_df['previous_cancellations']) == 0) & (train_df['adults'] ==0)
    train_df.loc[query, 'is_canceled'] = 1
    
###############################################################
# 強制置き換え test_df
# データに何らかの不備があると仮定して対処
###############################################################


###############################################################
# 強相関 train_df/test_df
###############################################################
# train/testで一致すると考えられるレコードに
# お互いのIDを付与したfeatureを作成する
# 限りなくリークっぽいけど。。
if 'is_ID_compare_match' in select_feature_categories:
    compare_taget_columns = [
        #'ID',
        'hotel',
        #'is_canceled',
        'lead_time',
        #'yyyy-mm-dd',
        #'arrival_date_year',
        #'arrival_date_month',
        #'arrival_date_week_number',
        #'arrival_date_day_of_month',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'adults',
        'children',
        'babies',
        'meal',
        'country',
        'market_segment',
        'distribution_channel',
        'is_repeated_guest',
        'previous_cancellations',
        'previous_bookings_not_canceled',
        'reserved_room_type',
        'assigned_room_type',
        'deposit_type',
        'agent',
        #'company',
        'days_in_waiting_list',
        'customer_type',
        'adr',
        'required_car_parking_spaces',
        'total_of_special_requests',
        #'yyyy-mm-dd_str',
        #'yyyy-mm-week',
        'yyyy-mm-week_UNIXTIME',
        ]
    
    # train/test一致比較？
    value_counts_result = pd.merge(train_df, test_df, on=compare_taget_columns, how='right')[['ID_x','ID_y']].value_counts().reset_index()

    # 辞書に変換
    train_result_dict = dict(zip(value_counts_result['ID_x'], value_counts_result['ID_y'].astype(int)))
    test_result_dict = dict(zip(value_counts_result['ID_y'], value_counts_result['ID_x'].astype(int)))

    # 一致したID番号列を生成 train/test双方に
    train_df['ID_compare_match'] = train_df['ID'].map(train_result_dict)
    train_df['ID_compare_match'] = train_df['ID_compare_match'].fillna(-1)
    train_df['ID_compare_match'] = train_df['ID_compare_match'].astype(int)
    
    test_df['ID_compare_match'] = test_df['ID'].map(test_result_dict)
    test_df['ID_compare_match'] = test_df['ID_compare_match'].fillna(-1)
    test_df['ID_compare_match'] = test_df['ID_compare_match'].astype(int)

###############################################################
# 異常を引き起こすfeatureを削ぎ落とす 
###############################################################
drop_columns = [
    #'ID',                                                                 # IDは評価には無関係
    #'adults',                                                             # SHAPでのマイナス貢献
    #'required_car_parking_spaces',                                        # SHAPでのマイナス貢献
    #'dev_value_counts_adults',                                            # SHAPでのマイナス貢献
    'relation_market_segment_and_customer_type',                          # 過学習
    'relation_distribution_channel_and_deposit_type',                     # 過学習
    'relation_deposit_type_and_customer_type',                            # 過学習
    'relation_market_segment_and_distribution_channel_and_deposit_type',  # 過学習
    'relation_market_segment_and_distribution_channel_and_customer_type', # やや悪化
    'domain_room_adr_per_nights',    # 悪化
    'domain_reservation_polularity', # 軽度の過学習
    'domain_diff_roomtype_from_reserved_room_type_to_assigned_room_type', # やや悪化
    'domain_low_cancel_rate_week',   # 過学習懸念
]

train_df = train_df.drop(columns=drop_columns)
test_df = test_df.drop(columns=drop_columns)

###############################################################
# datetime型,日付文字列型はlightGBMは受付NGのためdropする 
# 代わりにUNIXTIME,yyyy-mm-week_UNIXTIMEに頑張ってもらう
###############################################################
# train
train_df = train_df.drop(columns='yyyy-mm-dd')
train_df = train_df.drop(columns='yyyy-mm-dd_str')
train_df = train_df.drop(columns='yyyy-mm-week')

# test
test_df = test_df.drop(columns='yyyy-mm-dd')
test_df = test_df.drop(columns='yyyy-mm-dd_str')
test_df = test_df.drop(columns='yyyy-mm-week')

###############################################################
# データ確認 
###############################################################
pd.set_option('display.max_rows', None)
pprint(list(train_df.dtypes.index))
pprint(list(test_df.dtypes.index))
print(train_df.shape)
print(test_df.shape)

###############################################################
## pickle書き出し
###############################################################
train_df.to_pickle(f'{PREFIX_OUTPUT_DATA}/train_df_fix_feature.pickle')
test_df.to_pickle(f'{PREFIX_OUTPUT_DATA}/test_df_fix_feature.pickle')

## train/test 縦結合
train_test_df = pd.concat([train_df, test_df], axis=0)
train_test_df.to_pickle(f'{PREFIX_OUTPUT_DATA}/train_test_df_fix_feature.pickle')

################################################################
### pickle書き出し for gp
################################################################
#train_df.to_pickle(f'{PREFIX_OUTPUT_DATA}/train_df_gp_feature.pickle')
#test_df.to_pickle(f'{PREFIX_OUTPUT_DATA}/test_df_gp_feature.pickle')
#
### train/test 縦結合
#train_test_df = pd.concat([train_df, test_df], axis=0)
#train_test_df.to_pickle(f'{PREFIX_OUTPUT_DATA}/train_test_df_gp_feature.pickle')

