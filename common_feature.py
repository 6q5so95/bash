import math
import numpy as np
import os
import pandas as pd
import random
import torch
from scipy import stats
from scipy.stats import binom

import sklearn.preprocessing as preproc
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Optional, Tuple


# 実装方針
# 決まりきったパターンはdfを受け取って決めた名前と位置で属性追加して返す
#   Classに押し込める
# 何かと流動性要素のある属性はSeriesを受け取ってSeriesを返す、その後DataFrameへの組み込みは利用者で対応する
#   関数で個別に定義する

#################################################################
# class定義
#################################################################
# DataFrameを受け取り指定名でDataFrameに組み込んで返す


#################################################################
# 関数定義
#################################################################
def convert_to_categorical_features(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """DataFrameのfeatureを指定リストに基づきcategricalに変換する

    Args:
        df (pd.DataFrame): 変換対象DataFrame
        categorical_features (list): categorical 変換を指定したcolumns名リスト

    Returns:
        pd.DataFrame: 変換後DataFrame

    """
    # 対象columsをcategory属性に変換する
    for column in df.columns:
        if column in categorical_features:
            df[column] = df[column].astype('category')

    # assert
    for _ in df.columns:
        if _ in categorical_features:
            assert df[_].dtypes == 'category'

    return df


#################################################################
# 数学関数変換 
#################################################################
def create_stat_feature(df, cols: list):
    for col in cols:
        # min, max, std, mean, median
        new_colname = f'{col}_min_ratio'
        df[new_colname] = df[col].map(lambda x: x / (np.min(x) + 1e-8))
        new_colname = f'{col}_max_ratio'
        df[new_colname] = df[col].map(lambda x: x / (np.max(x) + 1e-8))
        new_colname = f'{col}_std_ratio'
        df[new_colname] = df[col].map(lambda x: x / (np.std(x) + 1e-8))
        new_colname = f'{col}_mean_ratio'
        df[new_colname] = df[col].map(lambda x: x / (np.mean(x) + 1e-8))
        new_colname = f'{col}_median_ratio'
        df[new_colname] = df[col].map(lambda x: x / (np.median(x) + 1e-8))
        
    return df


def create_math_feature(df: pd.DataFrame, target_key: str, target_columns: list)-> pd.DataFrame:
    # 数学関数属性追加
    ## 効果測定は苦労しそうな予感

    ## lead_time
    ## 作り出した属性と元のdfをIDをKeyにしてマージ
    #df = pd.merge(df, create_math_feature(df, 'lead_time', df_initial_columns), on='ID')

    _df = df.copy()

    #math_func = ['sum', 'mean', 'std', 'max', 'min']
    math_func = [np.sum, np.mean, np.max]

    # target_keyでgroupby
    agg_df = _df[target_columns].groupby([target_key], as_index=False).agg(math_func)

    # 生成したカラム名を適切な情報を持った名前に変更
    agg_df.columns = [
        #'_'.join(col) + f'_groupby_{target_key}' for col in agg_df.columns.values]
        f'math_' + '_'.join(col) + f'_groupby_{target_key}' for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)

    # ID、を残して他は作り出した属性だけのDataFrameを生成
    # agg_dfは target_keyと作り出した属性から構成されるDataFrameになっている
    # dfにあるtarget_keyとagg_dfのtarget_keyでleft join
    # つまりtarget_keyの値毎にdfへagg_df属性を横に追加する処理を行う
    merged_df = pd.merge(_df, agg_df, on=[target_key], how='left')
    
    # あんま関係なさそうなIDやUNIXTIMEの数学関数演算結果を外す
    return merged_df


##################################################
# 平均からの距離(ABS) 
##################################################
def create_feature_abs_mean(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series,pd.Series,pd.Series,pd.Series]:
    # lead_time
    # stays_in_weekend_nights
    # stays_in_week_nights
    # adults
    # adr
    # それぞれの平均からの乖離/abs
    mean_lead_time = df['lead_time'].mean()
    mean_stays_in_weekend_nights = df['stays_in_weekend_nights'].mean()
    mean_stays_in_week_nights = df['stays_in_week_nights'].mean()
    mean_adults = df['adults'].mean()
    mean_adr = df['adr'].mean()
    
    return \
        abs(df['lead_time'] - mean_lead_time), \
        abs(df['stays_in_weekend_nights'] - mean_stays_in_weekend_nights), \
        abs(df['stays_in_week_nights'] - mean_stays_in_week_nights), \
        abs(df['adults'] - mean_adults), \
        abs(df['adr'] - mean_adr)


##################################################
# clipping 
##################################################
def create_feature_clipped_only_upper(df: pd.DataFrame, target_col:str)->pd.Series:
    _df = df.copy()

    # 下限はそのまま、上限のみClipping
    p01 = df[target_col].quantile(0.000)
    p99 = df[target_col].quantile(0.999)

    return df[target_col].clip(p01, p99) # 上界1%をクリップ


def create_feature_clipped_lower_upper(df: pd.DataFrame, target_col:str)->pd.Series:
    _df = df.copy()

    # 下限上限でClipping
    p01 = df[target_col].quantile(0.001)
    p99 = df[target_col].quantile(0.999)

    return df[target_col].clip(p01, p99) # 上界・下界1%をクリップ


##################################################
# binning 
##################################################
def sturges_nbins(x):
    """
    スタージェスの公式からビンの数を算出する
    
    Parameters
    ----------
    x : array_like
        入力データ配列
    
    Returns
    ------- 
    nb : int
        ビンの数
    """

    n = len(x)
    range_x = max(x) - min(x)
    nb = 1 + math.ceil(math.log2(n)) + math.ceil(math.log2(1 + range_x / n))
    
    return nb


def create_feature_binning(df: pd.DataFrame, target_col: str, bins:int =10, labels:bool=False) -> pd.Series:
    # 指定featureに対してbinning区分を返す
    # ラベルは使用せず区分のみ利用
    # see. ラベルが必要な場合は labels=Falseを解除
    assert bins > 1
    _df = df.copy()
    
    return pd.cut(_df[target_col], bins=bins, labels=labels) 


def create_feature_binning_by_list(df:pd.DataFrame, target_col:str, bins: list)-> pd.Series:
    # bin区分を指定してbinning区分を返す
    _df = df.copy()
    
    # binning個別定義
    bins = bins

    return pd.cut(_df[target_col],bins=bins, right=False, labels=False)


def create_feature_binning_labels(df:pd.DataFrame, target_col:str, bins:int = 10)-> pd.Series:
    # ラベル区間の平均値→整数化して渡す
    _df = df.copy()
    
    cut = pd.cut(_df[target_col], bins=bins)
    cut = pd.Categorical(cut)
    
    return cut.map(lambda x: ((x.left + x.right)/2).astype(int))


def create_feature_binning_labels_counter(df: pd.DataFrame, target_col: str, bins: int) -> pd.DataFrame:
    """binラベル毎の特徴量を生成しレコード毎に所属するしないを0/1表現する

    Args:
        df (pd.DataFrame): 処理対象DataFrame
        target_col (str): df内のbin化対象column 
        num_bins (int): 分割数、スタージェスなどの公式にしたがい算出が望ましい

    Returns:
        pd.DataFrame: 元のDataFrameにbinラベルfeatureを追加した処理結果DataFrame 
    """
    _df = df.copy()

    # ビニング範囲の開始と終了値を計算
    min_value = _df[target_col].min()
    max_value = _df[target_col].max()
    bin_ranges = list(range(int(min_value), int(max_value) + 1, int((max_value - min_value) / bins)))

    # ビニングフラグを生成
    for bin_start, bin_end in zip(bin_ranges[:-1], bin_ranges[1:]):
        # bin_rangeを1つずらしてstart/endとして範囲比較を簡潔に書く
        bin_label = f'binning_labels_{target_col}_{bin_start}_{bin_end}'
        _df[bin_label] = ((_df[target_col] >= bin_start) & (_df[target_col] < bin_end)).astype(int)
        #_df[bin_label] = _df[bin_label].astype('category')

    return _df


##################################################
# value_counts() encoder 
##################################################
def create_feature_value_count_encoding(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    # 指定した属性に対してvalue_counts()を新たなfeatureとして追加する
    # mapなのでvalue_counts()結果のKeyとValueが！うまい具合にハマる！
    _df = df.copy()
    for target_col in target_cols:
        encoder = _df[target_col].value_counts()
        _df[f'dev_value_counts_{target_col}'] = _df[target_col].map(encoder)
    
    return _df


##################################################
# 日付系 
##################################################
def create_feature_is_weekend(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    # 週末フラグ
    query = ((_df['yyyy-mm-dd'].dt.weekday == 5) | (_df['yyyy-mm-dd'].dt.weekday == 6))
    _df.loc[query, 'is_weekend'] = 1
    _df['is_weekend'] = _df['is_weekend'].fillna(0)
    return _df['is_weekend']


def create_feature_seasons(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    return _df["arrival_date_month"].map({
    'March':     'Spring', 
    'April':     'Spring',
    'May':       'Spring',
    'June':      'Summer',
    'July':      'Summer',
    'August':    'Summer',
    'September': 'Autumn',
    'October':   'Autumn',
    'November':  'Autumn',
    'December':  'Winter',
    'January':   'Winter',
    'February':  'Winter',
    })


##################################################
# 分布標準化
##################################################
def create_feature_lead_time_to_standard_transform(df: pd.DataFrame) -> pd.Series:
    # lead_time分布をキレイに標準化
    # boxcoxとmimax_scaleの合わせ技
    return pd.Series(preproc.minmax_scale(stats.boxcox(df['lead_time'] + 0.01)[0]))


def create_feature_adr_to_standard_transform(df: pd.DataFrame) -> pd.Series:
    # adr分布をキレイに標準化
    # log10とmimax_scaleの合わせ技
    return pd.Series(preproc.minmax_scale(np.log10(df['adr']+1)))


def create_feature_agent_to_standard_transform(df: pd.DataFrame) -> pd.Series:
    # agent分布をキレイに標準化
    # mimax_scale
    return pd.Series(preproc.minmax_scale(df['agent']))


def create_feature_StanderdScaler(df: pd.DataFrame, target_colmns:list) -> pd.DataFrame:
    # StanderdScalerを指定Columnsに適用する
    _df = df.copy()
    ss = StandardScaler()
    for col in target_colmns:
        _df[f'ss_{col}'] = ss.fit_transform(_df[col].values.reshape(-1, 1))
    return _df


def create_feature_highvalue_features_sum_with_is_canceled(df: pd.DataFrame) -> pd.Series:
    # is_canceledと相関の高いfeatureをするという（StanderdScaler済（前提）
    _df = df.copy()
    return _df['ss_lead_time'] + _df['ss_country'] + _df['ss_deposit_type'] + _df['ss_distribution_channel']


def create_feature_lowvalue_features_sum_with_is_canceled(df: pd.DataFrame) -> pd.Series:
    # is_canceledと相関の低いfeatureをするという（StanderdScalor済（前提）
    _df = df.copy()
    return _df['ss_market_segment'] + _df['ss_lead_time'] + _df['ss_agent']


def create_feature_high_corrs_sum(df: pd.DataFrame) -> pd.Series:
    # is_canceledと相関が高いTop3をminmax_scalerして比率合算 0.52
    _df = df.copy()
    
    _df['standard_lead_time'] = pd.Series(preproc.minmax_scale(stats.boxcox(_df['lead_time'] + 0.01)[0]))
    _df['standard_country'] = pd.Series(preproc.minmax_scale(_df['country']))
    _df['standard_deposit_type'] = pd.Series(preproc.minmax_scale(_df['deposit_type']))
    _df['standard_distribution_channel'] = pd.Series(preproc.minmax_scale(_df['distribution_channel']))
    
    return _df['standard_lead_time']*0.9+ _df['standard_country']*0.5 + _df['standard_deposit_type'] * 1.2


##################################################
# relation系
# see. 000.EDA_【採用】feature複数の組み合わせに対するis_canceled指標の動向.ipynb
# - market_segment
# - deposit_type
# - customer_type
# - distribution_channel
##################################################
def create_feature_relation_market_segment_and_deposite_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    _df = df.copy()
    
    query1 = ((_df['market_segment'] == "Offline TA/TO") & (_df['deposit_type'] == "Non Refund")) 
    query2 = ((_df['market_segment'] == "Groups") & (_df['deposit_type'] == "Non Refund")) 
    query3 = ((_df['market_segment'] == "Online TA") & (_df['deposit_type'] == "Non Refund")) 
    query4 = ((_df['market_segment'] == "Corporate") & (_df['deposit_type'] == "Non Refund")) 

    # 初期値
    _df['relation_market_segment_and_deposite_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_market_segment_and_deposite_type'] = 1
    _df.loc[query2, 'relation_market_segment_and_deposite_type'] = 1
    _df.loc[query3, 'relation_market_segment_and_deposite_type'] = 1
    _df.loc[query4, 'relation_market_segment_and_deposite_type'] = 1

    return _df['relation_market_segment_and_deposite_type']


def create_feature_relation_market_segment_and_customer_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    _df = df.copy()

    query1 = ((_df['market_segment'] == "Groups") & (_df['customer_type'] == "Transient")) 
    query2 = ((_df['market_segment'] == "Groups") & (_df['customer_type'] == "Contract")) 

    # 初期値
    _df['relation_market_segment_and_customer_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_market_segment_and_customer_type'] = 1
    _df.loc[query2, 'relation_market_segment_and_customer_type'] = 1

    return _df['relation_market_segment_and_customer_type']


def create_feature_relation_distribution_channel_and_deposit_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    _df = df.copy()

    query1 = ((_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund")) 
    query2 = ((_df['distribution_channel'] == "Direct") & (_df['deposit_type'] == "Non Refund")) 
    query3 = ((_df['distribution_channel'] == "Corporate") & (_df['deposit_type'] == "Non Refund")) 

    # 初期値
    _df['relation_distribution_channel_and_deposit_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_distribution_channel_and_deposit_type'] = 1
    _df.loc[query2, 'relation_distribution_channel_and_deposit_type'] = 1
    _df.loc[query3, 'relation_distribution_channel_and_deposit_type'] = 1

    return _df['relation_distribution_channel_and_deposit_type']


def create_feature_relation_deposit_type_and_customer_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    _df = df.copy()

    query1 = ((_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Contract")) 
    query2 = ((_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query3 = ((_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient-Party")) 

    # 初期値
    _df['relation_deposit_type_and_customer_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_deposit_type_and_customer_type'] = 1
    _df.loc[query2, 'relation_deposit_type_and_customer_type'] = 1
    _df.loc[query3, 'relation_deposit_type_and_customer_type'] = 1

    return _df['relation_deposit_type_and_customer_type']


def create_feature_relation_market_segment_and_distribution_channel_and_deposit_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    _df = df.copy()
    
    query1 = ((_df['market_segment'] == "Corporate") & (_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund")) 
    query2 = ((_df['market_segment'] == "Offline TA/TO") & (_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund")) 
    query3 = ((_df['market_segment'] == "Groups") & (_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund")) 
    query4 = ((_df['market_segment'] == "Groups") & (_df['distribution_channel'] == "Direct") & (_df['deposit_type'] == "Non Refund")) 
    query5 = ((_df['market_segment'] == "Online TA") & (_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund")) 
    query6 = ((_df['market_segment'] == "Groups") & (_df['distribution_channel'] == "Corporate") & (_df['deposit_type'] == "Non Refund")) 
    query7 = ((_df['market_segment'] == "Corporate") & (_df['distribution_channel'] == "Corporate") & (_df['deposit_type'] == "Non Refund")) 

    # 初期値
    _df['relation_market_segment_and_distribution_channel_and_deposit_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1
    _df.loc[query2, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1
    _df.loc[query3, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1
    _df.loc[query4, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1
    _df.loc[query5, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1
    _df.loc[query6, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1
    _df.loc[query7, 'relation_market_segment_and_distribution_channel_and_deposit_type'] = 1

    return _df['relation_market_segment_and_distribution_channel_and_deposit_type']


def create_feature_relation_market_segment_and_distribution_channel_and_customer_type(df: pd.DataFrame) -> Tuple[pd.Series]:
    _df = df.copy()

    query1 = ((_df['market_segment'] == "Groups") & (_df['distribution_channel'] == "TA/TO") & (_df['customer_type'] == "Transient")) 

    # 初期値
    _df['relation_market_segment_and_distribution_channel_and_customer_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_market_segment_and_distribution_channel_and_customer_type'] = 1

    return _df['relation_market_segment_and_distribution_channel_and_customer_type']


def create_feature_relation_market_segment_and_deposit_type_and_customer_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    _df = df.copy()

    query1 = ((_df['market_segment'] == "Corporate") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query2 = ((_df['market_segment'] == "Offline TA/TO") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query3 = ((_df['market_segment'] == "Groups") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query4 = ((_df['market_segment'] == "Groups") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Contract")) 
    query5 = ((_df['market_segment'] == "Online TA") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query6 = ((_df['market_segment'] == "Offline TA/TO") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient-Party")) 

    # 初期値
    _df['relation_market_segment_and_deposit_type_and_customer_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_market_segment_and_deposit_type_and_customer_type'] = 1
    _df.loc[query2, 'relation_market_segment_and_deposit_type_and_customer_type'] = 1
    _df.loc[query3, 'relation_market_segment_and_deposit_type_and_customer_type'] = 1
    _df.loc[query4, 'relation_market_segment_and_deposit_type_and_customer_type'] = 1
    _df.loc[query5, 'relation_market_segment_and_deposit_type_and_customer_type'] = 1
    _df.loc[query6, 'relation_market_segment_and_deposit_type_and_customer_type'] = 1

    return _df['relation_market_segment_and_deposit_type_and_customer_type']


def create_feature_relation_distribution_channel_and_deposit_type_and_customer_type(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    _df = df.copy()

    query1 = ((_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query2 = ((_df['distribution_channel'] == "Corporate") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query3 = ((_df['distribution_channel'] == "Direct") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query4 = ((_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient")) 
    query5 = ((_df['distribution_channel'] == "TA/TO") & (_df['deposit_type'] == "Non Refund") & (_df['customer_type'] == "Transient-Party")) 

    # 初期値
    _df['relation_distribution_channel_and_deposit_type_and_customer_type'] = 0

    # 埋め込み
    _df.loc[query1, 'relation_distribution_channel_and_deposit_type_and_customer_type'] = 1
    _df.loc[query2, 'relation_distribution_channel_and_deposit_type_and_customer_type'] = 1
    _df.loc[query3, 'relation_distribution_channel_and_deposit_type_and_customer_type'] = 1
    _df.loc[query4, 'relation_distribution_channel_and_deposit_type_and_customer_type'] = 1
    _df.loc[query5, 'relation_distribution_channel_and_deposit_type_and_customer_type'] = 1

    return _df['relation_distribution_channel_and_deposit_type_and_customer_type']

##########################################################
# 平均予約数、平均滞在日数、平均価格、平均キャンセル率 
##########################################################
# 平均予約数
def create_group_ave_reserved_days(df: pd.DataFrame, list_features: list) -> pd.DataFrame:
    _df = df.copy()
    # 生成columns名構築
    columns = f"{'_'.join(list_features)}_ave_reserved_days"

    # groupy処理 
    result = _df.groupby(list_features).sum()['is_canceled'].reset_index()

    # 平均予約数算出
    count = len(_df)
    result['is_canceled'] = result['is_canceled']/count
    result = result.rename(columns={'is_canceled': columns})

    # 元dfへmerge
    _df = _df.merge(result, on=list_features, how='left')
    
    return _df

# 平均滞在日数
def create_group_ave_stay_days(df: pd.DataFrame, list_features: list) -> pd.DataFrame:
    _df = df.copy()
    # 生成columns名構築
    columns = f"{'_'.join(list_features)}_ave_reserved_days"
    
    # groupy処理 
    # 平均滞在日数算出
    days = ['stays_in_week_nights', 'stays_in_weekend_nights']
    result = _df.groupby(list_features)[days].mean()
    result[columns] = (result['stays_in_week_nights'] + result['stays_in_weekend_nights'])
    result = result[columns].reset_index()

    # merge
    _df = _df.merge(result, on=list_features, how='left')

    return _df

# 平均価格
def create_group_ave_adr(df: pd.DataFrame, list_features: list) -> pd.DataFrame:
    _df = df.copy()
    # 生成columns名構築
    columns = f"{'_'.join(list_features)}_ave_adr"

    # groupy処理 
    result = _df.groupby(list_features).mean()

    # 平均価格算出
    result = result['adr'].reset_index()
    result = result.rename(columns={'adr': columns})

    # merge
    _df = _df.merge(result, on=list_features, how='left')

    return _df

# キャンセル率
def create_group_ave_cancel_rate(df: pd.DataFrame, list_features: list) -> pd.DataFrame:
    _df = df.copy()
    # 生成columns名構築
    columns = f"{'_'.join(list_features)}_ave_cancel_rate"
    
    # groupy処理 
    result = _df.groupby(list_features).mean()

    # cancel rate算出
    result = result['is_canceled'].reset_index()
    result = result.rename(columns={'is_canceled': columns}) 

    # merge
    _df = _df.merge(result, on=list_features, how='left')

    return _df

##################################################
# 業務ドメイン分析
##################################################

##################################################
# 宿泊構成指標 
##################################################
def create_feature_family_size(df: pd.DataFrame) -> pd.Series:
    """ 大人＋子供＋幼児を合わせて家族人数を生成する

    Def:
        adult + children + babies

    Args:
        adults (pd.Series):   train/test['adult']
        children (pd.Series): train/test['children']
        babies (pd.Series):   train/test['babies']

    Returns:
        pd.Series: 家族数と推察できる合算値

    Example:
        df['family_size'] = create_feature_family_size(df)

    """
    _df = df.copy()
    return _df['adults'] + _df['children'] + _df['babies']


def create_feature_all_stay_days(df: pd.DataFrame) -> pd.Series:
    """平日・休日合算での総宿泊数

    Args:
        stays_in_weekend_nights (pd.Series): 週末（土曜または日曜）に滞在した宿泊数
        stays_in_week_nights (pd.Series):    滞在した週の夜数 (月曜から金曜)

    Returns:
        pd.Series: 週末平日を合算しての宿泊数（と解釈する）

    Example:
        df['all_stay_days'] = create_feature_all_stay_days(df['stays_in_weekend_nights'], df['stays_in_week_nigts'])

    """
    _df = df.copy()
    return _df['stays_in_weekend_nights'] + _df['stays_in_week_nights']


def create_feature_total_stay_costs(df: pd.DataFrame) -> pd.Series:
    # 宿泊予約あたりの総金額
    # 部屋平均金額☓宿泊総数
    ### 実際に予約のあったケースを算出
    _df = df.copy()
    return abs(_df['adr'] * (_df['stays_in_weekend_nights'] + _df['stays_in_week_nights']))
    #return abs(df['adr'] * (df['adults'] + df['children'])) * (~df['is_canceled'].fillna(0).astype(bool))


def create_feature_room_adr_per_nights(df: pd.DataFrame) -> pd.Series:
    # 宿泊予約の人単価平均金額
    # 子供料金は大人料金の80％と仮定する
    # 一人一泊あたりの平均料金を算出する
    _df = df.copy()
    CHILDREN_DISCOUNT_RATE = 0.80
    return abs((_df["adr"] / (_df["adults"] + _df["children"]/CHILDREN_DISCOUNT_RATE)).replace([np.inf, -np.inf], 0).fillna(0))


def create_feature_stay_adr_per_nights(df: pd.DataFrame) -> pd.Series:
    # 宿泊予約のあたり人単価平均金額
    # 子供料金は大人料金の80％と仮定する
    # 予約単位（宿泊日数込み）での一人あたり部屋料金
    _df = df.copy()
    CHILDREN_DISCOUNT_RATE = 0.80
    return abs((_df["adr"] * abs(_df['adr'] * (_df['stays_in_weekend_nights'] + _df['stays_in_week_nights']))) / (_df["adults"] + _df["children"]/CHILDREN_DISCOUNT_RATE)).replace([np.inf, -np.inf], 0).fillna(0)


#def create_feature_is_family(adults, children, babies):
#    ## 家族判定
#    ## ボツの予定
#    #df['is_family'] = df.apply(
#    #    lambda x: create_feature_is_family(x['adults'], x['children'], x['babies']),
#    #    axis=1,
#    #)
#    is_family = 0
#    if (adults >= 1) and (children >= 1 or babies >= 1):
#        is_family = 1
#    return is_family


def create_feature_travel_group(adults: int, children: int, babies: int) -> str:
    ## 大人子供幼児人数により分類
    # train_test_df['dev_travel_group'] = train_test_df.apply(
    #     lambda x: create_feature_travel_group(x['adults'], x['children'], x['babies']),
    #     axis=1,
    #     )

    if (adults == 1) and (children == 0) and (babies == 0):
        return 'Single'
    elif (adults == 2) and (children == 0) and (babies == 0):
        return 'Couple'
    elif (adults == 2) and ((children > 0) or (babies > 0)):
        return 'Family'
    elif (adults >= 3) and ((children == 0) and (babies == 0)):
        return 'Adult Event'
    elif (adults >= 3) and ((children > 0) or (babies > 0)):
        return 'Group'
    else:
        return 'Other'


##################################################
# 予約特性 
##################################################
def create_feature_is_deposit(deposit_type: pd.Series) -> pd.Series:
    # depositがあるかどうか（前払いの有無）
    if (deposit_type == 0) | ('deposit_type' == 2):
        return 0
    elif (deposit_type == 'No Deposit') | ('deposit_type' == 'Refundable'):
        return 0
    else:
        return 1


def create_feature_reservation_polularity(df: pd.DataFrame) -> pd.Series:
    # 予約人気度
    return (df['previous_cancellations'] + df['previous_bookings_not_canceled']).fillna(0)


def create_feature_diff_roomtype_from_reserved_room_type_to_assigned_room_type(df: pd.DataFrame) -> pd.Series:
    """予約に対する部屋タイプと実際に予約として割り当てられた部屋の差分を表現する

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    _df = df.copy()
    return df['reserved_room_type'] + df['assigned_room_type'] 


def create_feature_is_day_trip(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    # 日帰り予想フラグを生成する
    #query= (df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0) & (df['lead_time']==0) & (df['reserved_room_type']==0)
    query= (df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0) & (df['lead_time']==0) & (df['reserved_room_type']=='A')
    # 初期化
    _df['dev_is_day_trip'] = 0
    # day_trip判定追加
    _df.loc[query, 'dev_is_day_trip'] = 1
    
    return _df['dev_is_day_trip']

##################################################
# キャンセル率指標 
##################################################

## ！注意！
# is_canceledを使っての評価になっているのでリークリスクが高まります！
# 本当に使うかどうかは慎重に判断を要します

def create_feature_high_cancel_rate_week(df: pd.DataFrame)-> pd.Series:
    # 週単位でキャンセルレート高めをフラグ付け
    def create_feature_cancel_rate_week(x: str, check_list: list):
        if x in check_list:
            return 1
        else:
            return 0

    _df = df.copy()

    group1 = ['yyyy-mm-week']
    threshold = 0.45
    
    _df_week_calcel_rate = _df.groupby(group1, as_index=False).mean()[['yyyy-mm-week','is_canceled']]
    _list = list(_df_week_calcel_rate.query('is_canceled > @threshold')['yyyy-mm-week'])
    
    return _df['yyyy-mm-week'].apply(create_feature_cancel_rate_week, check_list=_list)


def create_feature_low_cancel_rate_week(df: pd.DataFrame)-> pd.Series:
    # 週単位でキャンセルレートの低めをフラグ付け
    def create_feature_cancel_rate_week(x: str, check_list: list):
        if x in check_list:
            return 1
        else:
            return 0

    _df = df.copy()

    group1 = ['yyyy-mm-week']
    threshold = 0.25
    
    _df_week_calcel_rate = _df.groupby(group1, as_index=False).mean()[['yyyy-mm-week','is_canceled']]
    _list = list(_df_week_calcel_rate.query('is_canceled < @threshold')['yyyy-mm-week'])
    
    return _df['yyyy-mm-week'].apply(create_feature_cancel_rate_week, check_list=_list)


#def create_feature_reservation_cancel_rate(df: pd.DataFrame) -> pd.Series:
#    ボツ -> previous_cancellationsが表現している
#
#    # 予約キャンセル発生割合
#    #return (df['previous_bookings_not_canceled']/(df['previous_cancellations'] + df['previous_bookings_not_canceled'])).replace([np.inf, -np.inf], 0).fillna(0)
#    return (df['previous_cancellations']/(df['previous_cancellations'] + df['previous_bookings_not_canceled'])).replace([np.inf, -np.inf], 0).fillna(0)


def create_feature_previous_booking_minus_canceled_and_notcanceled(df: pd.DataFrame) -> pd.Series:
    # キャンセルされていない予約数からキャンセルされた予約を減算
    ## 不正な大量予約が来ているのでは疑惑あり
    ## どうにも怪しい
    return df['previous_cancellations'] - df['previous_bookings_not_canceled']


def create_feature_cancel_rate_low_high(df: pd.DataFrame, cancel_rate: float, alpha=0.05)-> pd.DataFrame:
    # 日付単位でキャンセル率・二項分布から
    # キャンセル発生高い・低いの特異日傾向を還元する

    _df = df.copy()
    
    # yyyy-mm-dd_str日付単位でis_canceled情報を算出
    train_days = _df.groupby('yyyy-mm-dd_str')['is_canceled'].agg(['sum', 'size'])

    # キャンセル率が低い日を見つける
    p_low = binom.cdf(             # 2項分布
        k=train_days['sum'],       # k回以下の成功
        n=train_days['size'],      # 試行回数
        p=cancel_rate,             # キャンセル発生確率 
    )
    
    train_days['p_low'] = p_low
    #train_days['low'] = train_days['p_low'] <= alpha
    
    # キャンセル率が高い日を見つける
    p_high = 1 - binom.cdf(        # 2項分布
        k=train_days['sum']-1,     # k回以下の成功
        n=train_days['size'],      # 試行回数
        p=cancel_rate,             # キャンセル発生確率
    )
    train_days['p_high'] = p_high
    #train_days['high'] = train_days['p_high'] <= alpha
    
    train_days = train_days.reset_index()[['yyyy-mm-dd_str', 'p_low', 'p_high']]
    
    return pd.merge(_df, train_days, how='left', on='yyyy-mm-dd_str')


def create_feature_cancel_singularity_classification(x: float) -> float:
    # キャンセル発生特異日からlead_timeの区分けを行う
    # see. 001.EDA_【採用】lead_timeでキャンセルが急増するタイミング_キャンセル規約推察.ipynb
    #
    # df['dev_cancel_singularity_classification'] = df['lead_time'].map(create_feature_cancel_singularity_classification)
    
    range_intervals =[
        (0, 50),
    ]

    for i, (start, end) in enumerate(range_intervals):
        if start <= x < end:
            return 0 
        else:
            return 1 




#############################################################################
#############################################################################
# ここから先のfeature作成関数はis_canceled（目的変数）！を含んでいます！
# test_dfを含むDataFrameに適用してはいけません。リークします。
# train_df単独への適用＋cvなら許容範囲です

# - トレーニングデータとテストデータを厳密に分離する
# - CVの分割を大量にする
# - モデル間で特徴量を共有しない
# - テストデータからは特徴量を作らない
#############################################################################
#############################################################################

##################################################
# キャンセルレート（国別) 
# そもそもあれだけ偏った国別featureに判定の材料として適切なのか疑問がある
##################################################
#def create_feature_cancel_rate_category_by_country(df: pd.DataFrame, pickup_rate_threshold: int=10, classificate_category_size:int=10) -> pd.Series:
#    """ 国別のキャンセル発生率を算出しbinningしてレンジ分類する
#        ただし条件に該当しない国のキャンセル率は全体平均を設定する
#
#    Args:
#        df (pd.DataFrame): 評価対象DataFrame -> train_test_dfを想定
#        threshold (int, optional): 国別キャンセル発生率を拾う境界値 Defaults to 10.
#                                    評価レコードが少ないものは誤った判断となるリスクあり
#                                    まず一定のサンプル数が必要であり小さい母数は評価対象外
#        classificate_category (int, optional): カテゴリ分解数 Defaults to 10.
#
#    Returns:
#        pd.Series: 国別キャンセル発生率Category
#
#    Example:
#        df['cancel_rate_category_by_country'] = create_feature_cancel_rate_category_by_country(df, pickup_rate_threshold=10, classificate_category_size=10)
#
#    """
#    _df = df.copy()
#
#    # 国別にキャンセル発生数と国別予約数を算出してDataFrameへ埋め込み
#    _merge_df = pd.merge(
#        _df[_df['is_canceled'] == 1].groupby('country', as_index=False).size(),
#        _df.groupby('country', as_index=False).size(),
#        on='country',
#    )
#    # 国別キャンセル発生レート生成
#    _merge_df['cancel_rate_category_by_country'] = _merge_df['size_x'] / _merge_df['size_y']
#
#    # 国別キャンセルレート
#    ## 境界値で拾い上げ
#    cancel_rate_by_country_df = _merge_df.sort_values(by='cancel_rate_category_by_country', ascending=False).query('size_x > @pickup_rate_threshold')[['country', 'cancel_rate_category_by_country']]
#
#    # 本体へキャンセルレート追加
#    _df = pd.merge(_df, cancel_rate_by_country_df, on='country', how='left')
#    assert len(df) == len(_df)
#
#    # Nanレコードには全体平均を設定
#    # 閾値以下の国はtrain全体の平均キャンセル率で考えてみる、、か
#    _df['cancel_rate_category_by_country'] = _df['cancel_rate_category_by_country'].fillna(0.37041418879303123)
#
#    # あんまり精緻な数値だと過学習っぽくなりそうなので
#    # 少々大枠の区分分類で考えてみたいところ
#    cut_category = pd.cut(_df['cancel_rate_category_by_country'], classificate_category_size, duplicates='drop')
#
#    # 範囲の小さい側のみを取得する
#    # TODO label=Falseで試してみたい
#    return cut_category.apply(lambda x: x.left)
#
#
#def create_feature_country_cancel_rate(df: pd.DataFrame) -> pd.Series:
#    _df = df.copy()
#    # 国別キャンセルレート
#    # is_canceledが0 or 1であることを利用してキャンセル率を算出
#    # df = create_feature_country_cancel_rate(df)
#    country_cancel_rates = _df.groupby('country', as_index=False)['is_canceled'].mean().rename(columns={'is_canceled':'dev_country_cancel_rate'})
#
#    # merge
#    _df = _df.merge(country_cancel_rates, on='country', how='left')
#    
#    return _df['dev_country_cancel_rate']
#
#
#def create_feature_country_cancel_rate_by_yyyymm(df: pd.DataFrame)->pd.Series:
#    # 国別月別キャンセルレート
#    _df1 = df.copy() # 加工
#    _df2 = df.copy() # マスター
#
#    _df1['yyyy-mm'] = _df1['yyyy-mm-dd'].dt.strftime("%Y-%m")
#    _df2['yyyy-mm'] = _df2['yyyy-mm-dd'].dt.strftime("%Y-%m")
#    
#    _df1 = _df1.groupby(['yyyy-mm','country'], as_index=False).mean()[['yyyy-mm', 'country', 'is_canceled']].rename(columns={'is_canceled':'country_cancel_rate_yyyy-mm'})
#    _df2 = _df2.merge(_df1, on=['yyyy-mm','country'], how='left')
#    return _df2['country_cancel_rate_yyyy-mm']
#
#
#def create_feature_country_cancel_rate_by_yyyyweek(df: pd.DataFrame)->pd.Series:
#    # 国別週別キャンセルレート
#    _df1 = df.copy() # 加工
#    _df2 = df.copy() # マスター
#
#    _df1['yyyy-week'] = _df1['yyyy-mm-dd'].dt.strftime("%Y-%U")
#    _df2['yyyy-week'] = _df2['yyyy-mm-dd'].dt.strftime("%Y-%U")
#    
#    _df1 = _df1.groupby(['yyyy-week','country'], as_index=False).mean()[['yyyy-week', 'country', 'is_canceled']].rename(columns={'is_canceled':'country_cancel_rate_yyyy-week'})
#    _df2 = _df2.merge(_df1, on=['yyyy-week','country'], how='left')
#    return _df2['country_cancel_rate_yyyy-week']
#
#
#def create_feature_country_cancel_rate_by_yyyymmdd(df: pd.DataFrame)->pd.Series:
#    # 国別週別キャンセルレート
#    _df1 = df.copy() # 加工
#    _df2 = df.copy() # マスター
#    
#    _df1 = _df1.groupby(['yyyy-mm-dd','country'], as_index=False).mean()[['yyyy-mm-dd', 'country', 'is_canceled']].rename(columns={'is_canceled':'country_cancel_rate_yyyy-mm-dd'})
#    _df2 = _df2.merge(_df1, on=['yyyy-mm-dd','country'], how='left')
#    return _df2['country_cancel_rate_yyyy-mm-dd']
#
#
###################################################
## キャンセルレート（distribution_channel単独) 
###################################################
#def create_feature_cancel_rate_by_distribution_channel(df: pd.DataFrame) -> pd.Series:
#    # distribution_channel毎のキャンセル発生率をfeatureに組み込み
#    _df = df.copy()
#    # キャンセル率を計算
#    cancel_rates = _df.groupby('distribution_channel')['is_canceled'].mean().reset_index()
#    cancel_rates = cancel_rates.rename(columns={'is_canceled': 'distribution_channel_cancel_rate'})
#    _df = _df.merge(cancel_rates, on='distribution_channel', how='left')
#
#    return _df['distribution_channel_cancel_rate']
#
