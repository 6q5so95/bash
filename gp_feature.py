import pandas as pd
import numpy as np
import math

############################################################################
# random_state: 3133_X09
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/31/console
# add(add(sin(add(sin(add(X17, add(min(min(max(cos(X55), X133), max(cos(X27), X133)), add(min(X179, min(X255, X213)), X234)), add(min(log(X101), add(add(sin(add(X17, add(min(min(max(min(cos(sin(min(X255, X213))), log(X101)), X133), max(cos(X27), X133)), add(min(X179, add(min(X255, sin(X234)), X234)), X234)), add(X17, add(min(X179, X234), add(min(X179, cos(X27)), log(X265))))))), min(max(X179, log(X265)), X234)), X234)), log(X265))))), sin(add(X17, add(min(X179, X234), add(min(X179, min(cos(X27), add(min(X179, max(cos(X55), X133)), sin(X234)))), log(X265))))))), sin(sin(add(X17, add(min(max(X179, X133), X234), add(sin(min(X179, add(min(X255, sin(X234)), X234))), log(X265))))))), sin(sin(sin(sin(X234)))))
############################################################################
def create_gp_feature_3133_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X17 = _df['previous_cancellations']
    X55 = _df['arrival_date_monthmarket_segment_cat_combination_2']
    X133 = _df['binning_labels_lead_time_60_80']
    X27 = _df['total_of_special_requests']
    X179 = _df['starndard_lead_time']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X213 = _df['hotelyyyy-mm-week_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X101 = _df['assigned_room_typefeature_seasons_cat_combination_2']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    
    # 対策: NaNがある場合は0で埋める
    X133 = X133.fillna(0)
    X101 = X101.fillna(0)
    
    term1 = np.sin(
        np.minimum(
            np.sin(
                np.maximum(
                    X17,
                    np.minimum(
                        np.minimum(
                            np.maximum(np.cos(X55), X133),
                            np.maximum(np.cos(X27), X133)
                        ),
                        X179 + np.minimum(X255, X213) + X234
                    )
                )
            ),
            np.maximum(X17, X234)
        )
    )
    term2 = X17 + X234
    term3 = np.minimum(X179, np.minimum(X255, np.sin(X234))) + X234
    term4 = np.minimum(np.log(X101), term2 + term3 + np.sin(X17 + np.minimum(X179, X234) + np.minimum(X179, np.cos(X27) + np.log(X265))))
    term5 = np.minimum(np.minimum(X179, np.log(X265)), X234)
    term6 = X234 + term5
    term7 = np.log(X265)
    result = np.sin(term1 + term6) + np.sin(np.sin(X234))

    return pd.Series(result)


############################################################################
# random_state: 4900_X05
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/32/console
# add(cos(max(X221, max(min(div(X264, X130), sin(X234)), X192))), mul(neg(sin(div(div(X264, X130), X130))), min(X252, min(X234, add(X240, max(X240, mul(max(min(sin(div(X264, X130)), sin(X234)), X192), min(X252, min(X234, add(X240, min(div(X264, X130), min(div(X264, max(sin(X234), X192)), min(div(X264, X130), sin(X234))))))))))))))
############################################################################
def create_gp_feature_4900_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X221 = _df['hoteldeposit_type_cat_combination_2_target_encoder']
    X130 = _df['binning_labels_lead_time_0_20']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    
    term1 = np.maximum(
        X221,
        np.maximum(
            np.minimum(
                np.where(X130 != 0, np.divide(X264, X130), 0),  # Avoid division by zero
                np.sin(X234)
            ),
            X192
        )
    )
    
    inner_term = np.maximum(
        np.minimum(
            np.where(X130 != 0, np.divide(X264, X130), 0),  # Avoid division by zero
            np.sin(X234)
        ),
        X192
    )
    
    term2 = np.multiply(
        np.negative(
            np.sin(
                np.where(X130 != 0, np.divide(np.divide(X264, X130), X130), 0)  # Avoid division by zero
            )
        ),
        np.minimum(
            X252,
            np.minimum(
                X234,
                np.add(
                    X240,
                    np.maximum(
                        X240,
                        np.multiply(
                            np.maximum(
                                np.where(X130 != 0, np.sin(np.divide(X264, X130)), 0),  # Avoid division by zero
                                np.sin(X234)
                            ),
                            X192
                        ),
                        np.minimum(
                            X252,
                            np.minimum(
                                X234,
                                np.add(
                                    X240,
                                    np.minimum(
                                        np.divide(X264, X130),
                                        np.minimum(
                                            np.where(X130 != 0, np.divide(X264, np.maximum(np.sin(X234), X192)), 0),  # Avoid division by zero
                                            np.where(X130 != 0, np.divide(X264, X130), 0)  # Avoid division by zero
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    result = np.add(np.cos(term1), term2)

    return pd.Series(result)


############################################################################
# random_state: 4900_X09
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/32/console
# add(cos(max(X221, max(min(div(X264, X130), max(min(X240, max(X240, X22)), X192)), X192))), mul(neg(div(X264, X130)), min(X252, min(X234, add(X240, min(cos(min(X234, neg(sin(div(X264, X130))))), X240)))))) 
############################################################################
def create_gp_feature_4900_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X130 = _df['binning_labels_lead_time_0_20']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X22 =  _df['agent']
    X221 = _df['arrival_date_monthdeposit_type_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    
    
    # 対策: 分母が0の場合に0を返す
    term1 = np.cos(
        np.maximum(
            X221,
            np.maximum(
                np.minimum(
                    np.where(X130 == 0, 0, np.divide(X264, np.where(X130 == 0, 1, X130))),  # 0での除算を回避
                    np.maximum(
                        np.minimum(X240, np.maximum(X240, X22)),
                        X192
                    )
                ),
                X192
            )
        )
    )
    
    # 対策: 分母が0の場合に0を返す
    term2 = np.negative(np.where(X130 == 0, 0, np.divide(X264, np.where(X130 == 0, 1, X130))))  # 0での除算を回避
    
    # 対策: 分母が0の場合に1を返す
    term3 = np.minimum(
        X252,
        np.minimum(
            X234,
            np.add(
                X240,
                np.minimum(
                    np.cos(np.minimum(X234, np.negative(np.sin(np.where(X130 == 0, 0, np.divide(X264, np.where(X130 == 0, 1, X130))))))),  # 0での除算を回避
                    X240
                )
            )
        )
    )
    
    result = np.add(
        term1,
        np.multiply(term2, term3)
    )
    
    return pd.Series(result)


############################################################################
# random_state: 2693_X10
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/42/console
# neg(add(add(neg(neg(add(add(add(min(div(X234, add(X245, sub(X218, X130))), add(add(add(log(X265), add(X234, X17)), X239), X234)), X245), X192), add(min(div(add(add(X130, X17), X234), add(X17, X17)), add(add(add(add(log(X265), X239), X17), X234), X239)), X252)))), X192), add(add(cos(X27), min(X241, X72)), X252)))
############################################################################
def create_gp_feature_2693_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X218 = _df['arrival_date_monthdistribution_channel_cat_combination_2_target_encoder']
    X130 = _df['binning_labels_lead_time_0_20']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X241 = _df['countryfeature_seasons_cat_combination_2_target_encoder']
    X72  = _df['countrymarket_segment_cat_combination_2']
    
    term1 = np.negative(
        np.add(
            np.add(
                np.negative(
                    np.negative(
                        np.add(
                            np.add(
                                np.add(
                                    np.minimum(
                                        np.divide(
                                            X234,
                                            np.add(
                                                X245,
                                                np.subtract(X218, X130)
                                            )
                                        ),
                                        np.add(
                                            np.add(
                                                np.add(
                                                    np.log(X265),
                                                    np.add(X234, X17)
                                                ),
                                                X239
                                            ),
                                            X234
                                        )
                                    ),
                                    X245
                                ),
                                X192
                            ),
                            np.add(
                                np.minimum(
                                    np.divide(
                                        np.add(
                                            np.add(X130, X17),
                                            X234
                                        ),
                                        np.add(X17, X17)
                                    ),
                                    np.add(
                                        np.add(
                                            np.add(
                                                np.log(X265),
                                                X239
                                            ),
                                            X17
                                        ),
                                        X234
                                    )
                                ),
                                X252
                            )
                        )
                    )
                ),
                X192
            ),
            np.add(
                np.add(
                    np.cos(X27),
                    np.minimum(X241, X72)
                ),
                X252
            )
        )
    )
    
    return pd.Series(term1)

############################################################################
# random_state: 224_X08
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/57/console
# sub(sub(neg(min(X245, X240)), sin(sin(sin(sub(sub(sub(sub(sub(sin(sub(neg(sqrt(X234)), X234)), X265), min(min(sqrt(X234), X187), X179)), X265), max(inv(X110), cos(X27))), sqrt(min(X255, X240))))))), min(min(sqrt(X234), X187), X179))
############################################################################
def create_gp_feature_224_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy() 
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X187 = _df['doamin_stay_adr_per_nights']
    X179 = _df['starndard_lead_time']
    X110 = _df['math_lead_time_amax_groupby_yyyy-mm-week']
    X27 =  _df['total_of_special_requests']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    
    term1 = np.sqrt(X234)
    term2 = np.negative(term1)
    term3 = np.subtract(term2, X234)
    term4 = np.sin(term3)
    term5 = np.sin(term4)
    term6 = np.sin(term5)
    term7 = np.subtract(X265, term6)
    term8 = np.subtract(term7, X265)
    term9 = np.subtract(term8, np.minimum(np.minimum(term1, X187), X179))
    term10 = np.sin(term9)
    term11 = np.subtract(np.maximum(np.divide(1, X110), np.cos(X27)), term10)
    term12 = np.sqrt(np.minimum(X255, X240))
    term13 = np.subtract(np.negative(np.minimum(X245, X240)), term11)
    term14 = np.subtract(term13, term12)
    term15 = np.negative(np.minimum(term1, X187))
    term16 = np.minimum(term15, X179)
    term17 = np.minimum(term16, term14)
    
    return pd.Series(term17)


############################################################################
# random_state: 731_X10
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/60/console
# 731 add(min(add(min(X239, X255), min(min(X239, add(min(sin(X240), X179), sin(X240))), X255)), X179), add(min(min(X239, add(min(add(sin(X240), inv(X42)), X255), sin(add(min(min(sin(X17), X179), X255), X239)))), X255), add(sin(sub(min(X42, X255), add(add(add(add(X179, X234), sub(X255, neg(X245))), sin(X240)), sub(add(min(X239, X179), add(min(sin(X240), X179), inv(X42))), neg(X245))))), X245)))
############################################################################
def create_gp_feature_731_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X17 =  _df['previous_cancellations']
    X179 = _df['starndard_lead_time']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X42 =  _df['hotelarrival_date_month_cat_combination_2']
    
    term1 = np.minimum(X239, X255)
    term2 = np.minimum(np.minimum(X239, np.add(np.minimum(np.sin(X240), X179), np.sin(X240))), X255)
    term3 = np.add(term1, term2)
    term4 = np.add(term3, X179)
    term5 = np.add(np.add(np.add(X179, X234), np.subtract(X255, np.negative(X245))), np.sin(X240))
    #term6 = np.add(np.add(np.minimum(X239, np.add(np.minimum(np.add(np.sin(X240), np.reciprocal(X42)), X255), np.sin(np.add(np.minimum(np.minimum(np.sin(X17), X179), X255), X239)))), X255), np.add(np.minimum(np.sin(np.subtract(X42, X255))), X245))
    term6 = np.add(
        np.add(
            np.minimum(X239, np.add(np.minimum(np.add(np.sin(X240), np.reciprocal(X42)), X255), np.sin(np.add(np.minimum(np.minimum(np.sin(X17), X179), X255), X239)))),
            X255
        ),
        np.add(
            np.minimum(np.sin(np.subtract(X42, X255)), 0),
            X245
        )
    )
    term7 = np.subtract(np.minimum(X42, X255), term5)
    term8 = np.add(np.add(np.add(np.add(X179, X234), np.subtract(X255, np.negative(X245))), np.sin(X240)), np.subtract(np.add(np.minimum(X239, X179), np.add(np.minimum(np.sin(X240), X179), np.reciprocal(X42))), np.negative(X245)))
    term9 = np.sin(term7)
    term10 = np.add(term9, X245)
    term11 = np.add(term6, term10)
    term12 = np.add(term4, term11)
    
    return pd.Series(term12)

############################################################################
# random_state: 931_X09
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/107/console
# 931 add(min(min(tan(X195), inv(sqrt(sqrt(mul(min(min(X234, X32), X193), X262))))), add(add(X245, X239), sqrt(min(sqrt(mul(min(min(min(X234, X240), X193), X193), X262)), X239)))), add(add(X262, abs(add(add(neg(sqrt(X27)), X239), add(min(sqrt(sqrt(inv(sqrt(mul(min(X234, X240), X262))))), add(add(X245, X239), sqrt(min(X234, X32)))), add(add(X245, mul(X193, X262)), sqrt(min(X234, X32))))))), X245))
############################################################################
def create_gp_feature_931_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X193 = _df['p_low']
    X195 = _df['dev_cancel_singularity_classification']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X262 = _df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X32 =  _df['lead_time*adr']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X262 = _df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    
    term1 = np.tan(X195)
    term2 = np.sqrt(np.sqrt(np.multiply(np.minimum(np.minimum(X234, X32), X193), X262)))
    term3 = np.add(X245, X239)
    #term4 = np.minimum(np.sqrt(np.multiply(np.minimum(np.minimum(np.multiply(X234, X240), X193), X193), X262), X239))
    term4 =  np.minimum(np.sqrt(np.multiply(np.minimum(np.minimum(np.multiply(X234, X240), X193), X193), X262)), X239)
    term5 = np.subtract(np.sqrt(X27), X239)
    term6 = np.sqrt(np.multiply(np.minimum(np.multiply(X234, X240), X262), X32))
    term7 = np.add(X245, np.sqrt(np.minimum(X234, X32)))
    term8 = np.add(X245, np.multiply(X193, X262))
    term9 = np.sqrt(np.minimum(X234, X32))
    term10 = np.add(np.add(term5, X239), np.add(term6, np.add(term7, term9)))
    term11 = np.add(np.add(term4, term10), term8)
    result = np.add(np.minimum(term1, np.divide(1, term2)), np.add(term3, term11))
    
    return pd.Series(result)


############################################################################
# random_state: 4056_X07 
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/145/console
# 4056_X07 add(X139, add(add(add(add(X195, max(inv(X192), X245)), add(max(min(X239, X234), max(X192, X184)), add(add(max(X234, max(inv(X192), X234)), X239), add(max(max(max(X234, inv(X192)), X184), X184), X262)))), add(max(X192, X184), add(add(add(add(X245, X112), max(X262, X184)), X239), add(X192, max(max(add(X192, X262), X234), X184))))), X239))
############################################################################
def create_gp_feature_4056_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X139 = _df['binning_labels_lead_time_180_200']
    X195 = _df['dev_cancel_singularity_classification']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X262 = _df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X112 = _df['math_stays_in_weekend_nights_mean_groupby_yyyy-mm-week']
    
    term1 = X139
    term2 = X195 + np.maximum(1 / X192, X245)
    term3 = np.maximum(np.minimum(X239, X234), np.maximum(X192, X184))
    term4 = np.maximum(X234, 1 / X192)
    term5 = np.maximum(term4, X184)
    term6 = term1 + term2 + term3
    term7 = term6 + term5 + X262
    term8 = X192 + X184
    term9 = X245 + X112
    term10 = np.maximum(X262, X184)
    term11 = X239 + X192
    term12 = term8 + term9 + term10
    term13 = term7 + term11 + term12
    result = term13 + X239
    
    return pd.Series(result)


############################################################################
# random_state: 118_X05
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/159/console
# 118_N05(add(add(max(X264, add(max(X192, X265), max(X264, max(add(sin(add(max(X264, max(X192, add(inv(X60), X234))), X239)), X239), sin(add(add(sin(X234), neg(X138)), X239)))))), add(add(add(X192, add(add(X208, sin(add(X234, X234))), max(X155, add(max(X192, X265), X139)))), sin(add(add(X192, X234), X245))), add(max(X192, X265), add(add(X157, X255), sin(max(X192, X234)))))), sqrt(X240)), X195)
############################################################################
def create_gp_feature_118_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X60 =  _df['arrival_date_monthcustomer_type_cat_combination_2']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X138 = _df['binning_labels_lead_time_160_180']
    X208 = _df['hoteldistribution_channel_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X155 = _df['binning_labels_adr_156_182']
    X139 = _df['binning_labels_lead_time_180_200']
    X157 = _df['binning_labels_adr_208_234']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    
    term1_1 = np.maximum(X264, np.add(np.maximum(X192, X265), np.maximum(X264, np.maximum(np.add(np.sin(np.add(np.maximum(X264, np.maximum(X192, np.add(np.negative(X60), X234))), X239)), X239), np.sin(np.add(np.add(np.sin(X234), np.negative(X138)), X239))))))
    #term1_2 = np.add(np.add(np.add(X192, np.add(np.add(X208, np.sin(np.add(X234, X234))), np.maximum(X155, np.add(np.maximum(X192, X265), X139))), np.sin(np.add(np.add(X192, X234), X245))), np.add(np.maximum(X192, X265), np.add(np.add(X157, X255), np.sin(np.maximum(X192, X234))))))
    #term1_2 = np.add(np.add(np.add(X192, np.add(np.add(X208, np.sin(np.add(X234, X234))), np.maximum(X155, np.add(np.maximum(X192, X265), X139))), np.sin(np.add(np.add(X192, X234), X245))), np.add(np.maximum(X192, X265), np.add(np.add(X157, X255), np.sin(np.maximum(X192, X234))))
    term1_2 = np.add(
        np.add(
            np.add(
                X192,
                np.add(
                    np.add(
                        X208,
                        np.sin(np.add(X234, X234))
                    ),
                    np.maximum(X155, np.add(np.maximum(X192, X265), X139))
                )
            ),
            np.sin(np.add(np.add(X192, X234), X245))
        ),
        np.add(np.maximum(X192, X265), np.add(np.add(X157, X255), np.sin(np.maximum(X192, X234))))
    )
    
    term1_3 = np.sqrt(X240)
    term1 = np.add(term1_1, term1_2, term1_3)
    result = np.add(term1, X195)

    return pd.Series(result)


############################################################################
# random_state: 276_X05
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/174/console
# 276_X05: sin(sin(sin(sub(cos(sin(log(X255))), sub(sin(add(X265, sin(sin(sub(sin(sin(X234)), sub(cos(add(X264, X234)), sin(sin(X179)))))))), sub(cos(sin(add(add(sin(X234), add(cos(abs(log(X255))), mul(X27, sin(sin(X179))))), sin(add(add(sin(sub(add(sin(sin(X179)), sin(X234)), sub(X234, sin(sin(X179))))), sin(X234)), abs(log(X255))))))), sin(sin(add(X264, add(neg(X26), sin(add(add(sin(X234), sin(X234)), X234))))))))))))
############################################################################
def create_gp_feature_276_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X179 = _df['starndard_lead_time']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X27 =  _df['total_of_special_requests']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    
    term1 = np.sin(
        np.sin(
            np.sin(
                np.subtract(
                    np.cos(
                        np.sin(
                            np.log(X255)
                        )
                    ),
                    np.subtract(
                        np.sin(
                            np.add(
                                X265,
                                np.sin(
                                    np.sin(
                                        np.subtract(
                                            np.sin(
                                                np.subtract(
                                                    np.cos(
                                                        np.add(
                                                            X264,
                                                            X234
                                                        )
                                                    ),
                                                    np.sin(
                                                        np.sin(
                                                            X179
                                                        )
                                                    )
                                                )
                                            ),
                                            np.sin(
                                                np.add(
                                                    np.add(
                                                        np.sin(X234),
                                                        np.add(
                                                            np.cos(
                                                                np.abs(
                                                                    np.log(X255)
                                                                )
                                                            ),
                                                            np.multiply(
                                                                X27,
                                                                np.sin(
                                                                    np.sin(X179)
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    np.sin(
                                                        np.add(
                                                            np.add(
                                                                np.sin(
                                                                    np.subtract(
                                                                        np.add(
                                                                            np.sin(
                                                                                np.sin(X179)
                                                                            ),
                                                                            np.sin(X234)
                                                                        ),
                                                                        np.sin(
                                                                            np.sin(X179)
                                                                        )
                                                                    )
                                                                ),
                                                                np.sin(X234)
                                                            ),
                                                            np.abs(
                                                                np.log(X255)
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        np.sin(
                            np.sin(
                                np.add(
                                    X264,
                                    np.add(
                                        np.negative(X26),
                                        np.sin(
                                            np.add(
                                                np.add(
                                                    np.sin(X234),
                                                    np.sin(X234)
                                                ),
                                                X234
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    
    return pd.Series(term1)

############################################################################
# random_state: 4686_X07
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/183/console
# sub(min(sqrt(sqrt(X240)), mul(tan(cos(X26)), max(X184, mul(cos(X26), min(mul(tan(cos(X130)), min(max(max(cos(X26), tan(cos(X26))), cos(X27)), min(tan(X255), X234))), max(abs(X184), mul(min(X41, cos(X27)), min(mul(tan(cos(X130)), min(tan(X255), X234)), min(X41, max(min(mul(sqrt(X240), min(tan(X255), X234)), mul(tan(X255), min(mul(tan(X255), min(sqrt(sqrt(X240)), X234)), min(X41, sqrt(X240))))), mul(min(X41, cos(X27)), min(mul(tan(cos(X130)), min(tan(X255), X234)), sqrt(cos(X264)))))))))))))), cos(X264))
############################################################################
def create_gp_feature_4686_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X130 = _df['binning_labels_lead_time_0_20']
    X27 =  _df['total_of_special_requests']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X41 =  _df['adr*arrival_date_week_number*arrival_date_day_of_month']
    X42 =  _df['hotelarrival_date_month_cat_combination_2']
    X264 = _df['binning_labels_adr_390_416']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X36  = _df['adr*arrival_date_day_of_month']
    X179 = _df['starndard_lead_time']
    X137 = _df['binning_labels_lead_time_140_160']
    
    term1 = np.subtract(
        np.multiply(
            np.sin(
                np.multiply(
                    np.subtract(
                        X234,
                        np.minimum(
                            np.minimum(
                                np.cos(X42),
                                X265
                            ),
                            X239
                        )
                    ),
                    np.subtract(
                        np.tan(
                            np.divide(
                                np.subtract(
                                    np.cos(
                                        np.minimum(
                                            X36,
                                            X239
                                        )
                                    ),
                                    X265
                                ),
                                X239
                            )
                        ),
                        np.subtract(
                            X27,
                            X265
                        )
                    )
                )
            ),
            np.subtract(
                np.subtract(
                    np.sin(
                        np.add(
                            X234,
                            X234
                        )
                    ),
                    np.negative(
                        np.minimum(
                            X255,
                            np.add(
                                X234,
                                X234
                            )
                        )
                    )
                ),
                np.multiply(
                    np.tan(X179),
                    X137
                )
            )
        ),
        np.subtract(
            np.subtract(
                np.sin(
                    np.add(
                        X234,
                        X234
                    )
                ),
                np.negative(X255)
            ),
            np.multiply(
                np.multiply(
                    X239,
                    X137
                ),
                X137
            )
        )
    )
    
    result = term1

    return pd.Series(result)


############################################################################
# random_state: 4389_X03
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/187/console
# max(X192, max(X182, min(sub(add(X254, X254), tan(X26)), min(mul(add(X195, X234), sqrt(sqrt(min(X239, min(min(X187, min(X234, X255)), X234))))), div(div(min(X239, min(X187, min(X234, X255))), sub(add(X250, X270), X192)), sub(add(X250, X219), X192))))))
############################################################################
def create_gp_feature_4389_X03(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X182 = _df['relation_market_segment_and_deposite_type']
    X254 = _df['distribution_channelfeature_seasons_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X195 = _df['dev_cancel_singularity_classification']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X187 = _df['doamin_stay_adr_per_nights']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X250 = _df['distribution_channelassigned_room_type_cat_combination_2_target_encoder']
    X270 = _df['domain_travel_group_target_encoder']
    X219 = _df['arrival_date_monthreserved_room_type_cat_combination_2_target_encoder']
    
    term1 = np.maximum(
        np.maximum(
            X192,
            X182
        ),
        np.minimum(
            (X254 + X254) - np.tan(X26),
            np.minimum(
                (X195 + X234) * np.sqrt(np.sqrt(np.minimum(X239, np.minimum(np.minimum(X187, np.minimum(X234, X255)), X234)))),
                np.divide(np.divide(np.minimum(X239, np.minimum(X187, np.minimum(X234, X255))),
                            (X250 + X270) - X192),
                (X250 + X219) - X192)
            )
        )
    )
    result = term1
    
    return pd.Series(result)


############################################################################
# random_state: 800_X04
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/199/console
# 800_X04
# add(add(min(X234, mul(X179, add(min(X240, min(add(mul(X179, X22), X238), mul(min(min(X245, abs(X265)), abs(X47)), add(abs(X265), X245)))), min(X196, X255)))), X265), sin(add(min(X240, mul(X179, add(X234, min(X234, X255)))), max(X157, add(min(X234, mul(X179, add(min(X245, min(X196, X255)), min(add(neg(X27), X265), X255)))), X238)))))
############################################################################
def create_gp_feature_800_X04(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X22 =  _df['agent']
    X238 = _df['countrydeposit_type_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X47 =  _df['hotelreserved_room_type_cat_combination_2']
    X196 = _df['hotel_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X157 = _df['binning_labels_adr_208_234']
    X27 =  _df['total_of_special_requests']
    
    term1 = (
        np.minimum(X234, np.multiply(X179, np.minimum(
            X240, np.minimum(
                np.add(np.multiply(X179, X22), X238),
                np.multiply(
                    np.minimum(np.minimum(X245, np.abs(X265)), np.abs(X47)),
                    np.add(np.abs(X265), X245)
                )
            )
        )))
        + X265 * np.sin(np.minimum(X240, np.multiply(X179, np.add(
            X234, np.minimum(X234, X255)
        ))) + np.maximum(
            X157, np.add(np.minimum(X234, np.multiply(X179, np.add(
                np.minimum(X245, np.minimum(X196, X255)),
                np.minimum(-X27, X265)
            ))), X238)
        ))
    )
    result = term1
    return pd.Series(result)


############################################################################
# random_state: 4861_X09 
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/229/console
# 4861_X09
# max(max(max(max(sin(X170), max(X192, X264)), sin(sin(sin(add(X234, max(X192, X264)))))), sin(X170)), max(X192, X201))
############################################################################
def create_gp_feature_4861_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X170 = _df['dev_value_counts_adults']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X201 = _df['deposit_type_target_encoder']
    
    term1 = np.maximum(np.maximum(np.maximum(np.sin(X170), np.maximum(X192, X264)),
                        np.sin(np.sin(np.sin(np.add(X234, np.maximum(X192, X264))))), np.sin(X170)), X192)
    
    result = term1
    return pd.Series(result)


############################################################################
# random_state: 4083_X06
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/240/console
# 4083_X06
# add(add(min(add(X179, min(div(X155, X27), min(X179, inv(add(X234, X265))))), add(X234, min(add(X179, min(inv(min(inv(X240), max(inv(inv(X240)), add(X234, X265)))), X179)), add(X234, add(X234, X265))))), add(add(tan(inv(min(inv(X240), max(inv(min(add(X179, X234), add(X240, X234))), add(X234, X234))))), X192), X265)), X234)
############################################################################
def create_gp_feature_4083_X06(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X27 =  _df['total_of_special_requests']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X155 = _df['binning_labels_adr_156_182']
    
    term1 = np.add(
        np.add(
            np.add(
                np.add(
                    X179,
                    np.minimum(
                        np.add(
                            np.where(X27 != 0, np.divide(X155, X27), 0),
                            np.add(X179, np.divide(1, np.where(X240 != 0, np.divide(1, X240), 0)))
                        ),
                        X234
                    )
                ),
                np.add(
                    np.tan(
                        np.where(
                            X240 != 0,
                            np.minimum(
                                np.divide(1, np.where(X240 != 0, np.minimum(
                                    np.add(X179, X234),
                                    np.add(X240, X234)
                                ), 0)),
                                X234
                            ),
                            0
                        )
                    ),
                    X192
                )
            ),
            X265
        ),
        X234
    )
    result = term1
    return pd.Series(result)

############################################################################
# random_state: 4083_X07
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/240/console
# 4083_X07
# add(add(min(div(X155, X27), add(X234, add(X234, X265))), add(add(tan(inv(min(inv(X240), max(inv(min(add(X179, X234), add(add(X234, X265), X234))), add(X234, X234))))), X192), add(X234, X265))), X265)
############################################################################
def create_gp_feature_4083_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X27 =  _df['total_of_special_requests']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X155 = _df['binning_labels_adr_156_182']

    term1 = np.add(
        np.add(
            np.add(
                np.add(
                    X179,
                    np.minimum(
                        np.add(
                            np.where(X27 != 0, np.divide(X155, X27), 0),
                            np.add(
                                X234,
                                np.add(X234, X265)
                            )
                        ),
                        X234
                    )
                ),
                np.add(
                    np.tan(
                        np.where(
                            X240 != 0,
                            np.minimum(
                                np.where(X240 != 0, np.divide(1, X240), 0),
                                X234
                            ),
                            0
                        )
                    ),
                    X192
                )
            ),
            X265
        ),
        X265
    )
    result = term1
    
    return pd.Series(result)


############################################################################
# random_state: 3168_X06
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/254/console
# 3168_X06
# add(add(cos(max(X255, X234)), add(cos(cos(log(X27))), add(cos(cos(cos(inv(abs(X255))))), add(cos(cos(cos(inv(abs(X27))))), max(sub(X265, neg(max(X182, X234))), sub(X192, neg(X192))))))), add(cos(max(log(X27), X234)), add(cos(sin(X26)), max(cos(cos(cos(cos(X234)))), add(add(X234, sub(X265, X182)), sqrt(X255))))))
############################################################################
def create_gp_feature_3168_X06(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X182 = _df['relation_market_segment_and_deposite_type']
    X26 =  _df['required_car_parking_spaces']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']

    # ゼロ割やNaNを防ぐために分母が0またはNaNの場合は計算結果を0に設定
    def safe_divide(a, b):
        return np.divide(a, np.where((b == 0) | np.isnan(b), 1, b))

    term1 = np.add(
        np.add(
            np.cos(
                np.maximum(X255, X234)
            ),
            np.add(
                np.cos(
                    np.cos(
                        np.log(np.where(X27 != 0, X27, 1))
                    )
                ),
                np.add(
                    np.cos(
                        np.cos(
                            np.cos(
                                np.abs(safe_divide(1, X255))
                            )
                        )
                    ),
                    np.add(
                        np.cos(
                            np.cos(
                                np.cos(
                                    safe_divide(1, X27)
                                )
                            )
                        ),
                        np.maximum(
                            np.subtract(X265, np.negative(np.maximum(X182, X234))),
                            np.subtract(X192, np.negative(X192))
                        )
                    )
                )
            )
        ),
        np.add(
            np.cos(
                np.maximum(np.log(np.where(X27 != 0, X27, 1)), X234)
            ),
            np.add(
                np.cos(
                    np.sin(X26)
                ),
                np.maximum(
                    np.cos(
                        np.cos(
                            np.cos(
                                np.cos(X234)
                            )
                        )
                    ),
                    np.add(
                        np.add(
                            X234,
                            np.subtract(X265, X182)
                        ),
                        np.sqrt(X255)
                    )
                )
            )
        )
    )
    result = term1

    return pd.Series(result)


############################################################################
# random_state: 4625_X01
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/256/console
# 4625_X01
# add(add(X265, add(add(add(X265, abs(abs(add(abs(min(X234, X240)), neg(X130))))), neg(X130)), abs(abs(sub(min(X234, min(X234, X187)), X189))))), min(sqrt(X255), add(min(min(abs(add(min(X234, sqrt(X255)), neg(sqrt(X255)))), X234), X240), min(X234, add(div(cos(X118), min(X74, X27)), min(X234, X240))))))
############################################################################
def create_gp_feature_4625_X01(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X130 = _df['binning_labels_lead_time_0_20']
    X187 = _df['doamin_stay_adr_per_nights']
    X189 = _df['domain_is_deposit']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X118 = _df['math_adults_mean_groupby_yyyy-mm-week']
    X74 =  _df['countryreserved_room_type_cat_combination_2']
    X27 =  _df['total_of_special_requests']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.add(
        X265,
        np.add(
            np.add(
                np.add(
                    X265,
                    np.abs(
                        np.abs(
                            np.add(
                                np.abs(
                                    np.minimum(X234, X240)
                                ),
                                np.negative(X130)
                            )
                        )
                    )
                ),
                np.negative(X130)
            ),
            np.abs(
                np.abs(
                    np.subtract(
                        np.minimum(X234, np.minimum(X234, X187)),
                        X189
                    )
                )
            )
        )
    )

    term2 = np.minimum(
        np.sqrt(X255),
        np.add(
            np.minimum(
                np.minimum(
                    np.abs(
                        np.add(
                            np.minimum(X234, np.sqrt(X255)),
                            np.negative(np.sqrt(X255))
                        )
                    ),
                    X234
                ),
                X240
            ),
            np.minimum(
                X234,
                np.add(
                    np.divide(
                        np.cos(X118),
                        np.minimum(X74, np.where(X27 != 0, X27, 1))
                    ),
                    np.minimum(X234, X240)
                )
            )
        )
    )

    result = np.add(term1, term2)

    # -np.infを0に置き換え
    result = pd.Series(np.where(np.isneginf(result), 0, result))

    return pd.Series(result)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/259/console
# 4981_X02
# add(add(add(add(X265, add(min(X179, add(X257, min(cos(tan(X27)), add(min(min(X179, min(X234, X268)), min(X66, cos(X27))), max(sqrt(X17), min(X179, X234)))))), max(X184, min(cos(tan(X27)), add(min(min(X179, X234), min(X66, min(min(X179, X234), X268))), max(sqrt(X17), min(X179, X234))))))), min(min(X179, min(X234, X268)), X268)), min(X179, add(X257, max(X17, X234)))), X240)
############################################################################
def create_gp_feature_4981_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X257 = _df['reserved_room_typecustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X268 = _df['customer_typefeature_seasons_cat_combination_2_target_encoder']
    X66 =  _df['mealreserved_room_type_cat_combination_2']
    X17 =  _df['previous_cancellations']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = (
        X265 +
        np.add(
            np.minimum(X179,
                X257 + np.minimum(
                    np.cos(np.tan(X27)),
                    np.minimum(
                        np.minimum(X179, X234),
                        X268
                    )
                )
            ),
            np.maximum(X184,
                np.minimum(
                    np.cos(np.tan(X27)),
                    np.minimum(
                        np.minimum(X179, X234),
                        np.minimum(
                            np.minimum(X66,
                                np.minimum(
                                    np.minimum(X179, X234),
                                    X268
                                )
                            ),
                            np.maximum(np.sqrt(X17), np.minimum(X179, X234))
                        )
                    )
                )
            )
        ) +
        np.minimum(
            np.minimum(X179, X234, X268),
            X268
        ) +
        np.minimum(X179, X257 + np.maximum(X17, X234)) +
        X240
    )

    # -np.infを0に置き換え
    term1 = pd.Series(np.where(np.isneginf(term1), 0, term1))
    
    result = term1

    return pd.Series(result)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/259/console
# 4981_X09
# add(add(add(X265, add(min(X179, add(X257, min(X234, X268))), max(X184, min(cos(tan(X27)), add(min(min(X179, add(X234, add(min(tan(X27), X234), X234))), min(X66, cos(X27))), max(sqrt(X17), X17)))))), min(min(X179, X234), X268)), X240)
############################################################################
def create_gp_feature_4981_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X257 = _df['reserved_room_typecustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X234 = _df['lead_time*arrival_date_day_of_month']
    X268 = _df['customer_typefeature_seasons_cat_combination_2_target_encoder']
    X66 =  _df['mealreserved_room_type_cat_combination_2']
    X17 =  _df['previous_cancellations']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = (
        X265 +
        np.add(
            np.add(
                np.add(
                    X179,
                    np.add(
                        X257,
                        np.minimum(
                            X234,
                            X268
                        )
                    )
                ),
                np.maximum(
                    X184,
                    np.minimum(
                        np.cos(np.tan(X27)),
                        np.add(
                            np.add(
                                np.minimum(
                                    np.minimum(X179, X234),
                                    np.add(
                                        X234,
                                        np.add(
                                            np.minimum(np.tan(X27), X234),
                                            X234
                                        )
                                    )
                                ),
                                np.minimum(X66, np.cos(X27))
                            ),
                            np.maximum(np.sqrt(X17), X17)
                        )
                    )
                )
            ),
            np.minimum(
                np.minimum(X179, X234),
                X268
            )
        ) +
        X240
    )

    # ゼロ割および np.inf, -np.inf を回避する
    term1 = pd.Series(np.where(np.isinf(term1) | np.isnan(term1), 0, term1))

    return pd.Series(term1)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/259/console
# 4981_X07
# add(add(add(X265, add(min(X179, add(X234, max(X17, X234))), max(X184, min(cos(tan(X27)), add(min(min(X179, X234), min(X66, cos(X27))), max(min(X234, X268), min(X179, X234))))))), min(min(X179, X234), X268)), X240)
############################################################################
def create_gp_feature_4981_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X257 = _df['reserved_room_typecustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X234 = _df['lead_time*arrival_date_day_of_month']
    X268 = _df['customer_typefeature_seasons_cat_combination_2_target_encoder']
    X66 =  _df['mealreserved_room_type_cat_combination_2']
    X17 =  _df['previous_cancellations']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = (
        X265 +
        np.add(
            np.add(
                np.add(
                    X179,
                    np.add(
                        X234,
                        np.maximum(
                            X17,
                            X234
                        )
                    )
                ),
                np.maximum(
                    X184,
                    np.minimum(
                        np.cos(np.tan(X27)),
                        np.add(
                            np.add(
                                np.minimum(
                                    np.minimum(X179, X234),
                                    np.add(
                                        X66,
                                        np.cos(X27)
                                    )
                                ),
                                np.maximum(
                                    np.minimum(X234, X268),
                                    np.minimum(X179, X234)
                                )
                            ),
                            0  # ゼロ割防止
                        )
                    )
                )
            ),
            np.minimum(
                np.minimum(X179, X234),
                X268
            )
        ) +
        X240
    )

    return pd.Series(term1)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/332/console
# 3471_X10
# add(add(abs(add(X192, add(X212, add(sin(abs(add(add(X234, X192), min(sin(sin(X255)), abs(X264))))), sin(add(add(X234, X192), abs(X264))))))), add(min(sin(abs(X264)), sin(sin(X255))), X221)), min(X195, min(sin(abs(min(X216, sin(sin(X255))))), add(min(sin(X234), sin(X255)), X221))))
############################################################################
def create_gp_feature_3471_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X212 = _df['hotelcustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X221 = _df['arrival_date_monthdeposit_type_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X216 = _df['arrival_date_monthcountry_cat_combination_2_target_encoder']

    term1 = np.add(
        np.abs(
            np.add(X192,
                np.add(X212,
                    np.add(
                        np.sin(
                            np.abs(
                                np.add(
                                    np.add(X234, X192),
                                    np.minimum(
                                        np.sin(
                                            np.sin(X255)
                                        ),
                                        np.abs(X264)
                                    )
                                )
                            )
                        ),
                        np.sin(
                            np.add(
                                np.add(X234, X192),
                                np.abs(X264)
                            )
                        )
                    )
                )
            )
        ),
        np.minimum(
            np.sin(
                np.abs(X264)
            ),
            np.sin(
                np.sin(X255)
            )
        )
    )

    term2 = np.add(
        np.minimum(
            X195,
            np.minimum(
                np.sin(
                    np.abs(
                        np.minimum(
                            X216,
                            np.sin(
                                np.sin(X255)
                            )
                        )
                    )
                ),
                np.add(
                    np.minimum(
                        np.sin(X234),
                        np.sin(X255)
                    ),
                    X221
                )
            )
        ),
        np.minimum(
            X195,
            np.minimum(
                np.sin(
                    np.abs(
                        np.minimum(
                            X216,
                            np.sin(
                                np.sin(X255)
                            )
                        )
                    )
                ),
                np.add(
                    np.minimum(
                        np.sin(X234),
                        np.sin(X255)
                    ),
                    X221
                )
            )
        )
    )

    result = np.add(
        term1,
        term2
    )

    return pd.Series(result)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/349/console
# 3643_X05
# add(add(add(add(add(inv(X180), min(X240, sqrt(sqrt(min(X193, X255))))), sub(min(add(X179, X149), sqrt(min(X240, X234))), X26)), add(X255, max(X234, X17))), min(sqrt(min(X240, X234)), X94)), add(max(sub(add(X255, max(X234, X17)), X26), max(X221, X17)), min(sqrt(X239), add(X179, X149))))
############################################################################
def create_gp_feature_3643_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X180 = _df['starndard_adr']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X193 = _df['p_low']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X149 = _df['binning_labels_adr_0_26']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X17 =  _df['previous_cancellations']
    X94 =  _df['reserved_room_typedeposit_type_cat_combination_2']
    X221 = _df['arrival_date_monthdeposit_type_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']

    term1 = (
        np.add(
            np.add(
                np.add(
                    np.add(
                        np.divide(1, X180),
                        np.minimum(X240, np.sqrt(np.sqrt(np.minimum(X193, X255)))
                        )),
                    np.subtract(
                        np.minimum(
                            np.add(X179, X149),
                            np.sqrt(np.minimum(X240, X234))
                        ),
                        X26
                    )
                ),
                np.add(X255, np.maximum(X234, X17))
            ),
            np.minimum(np.sqrt(np.minimum(X240, X234)), X94)
        )
    )

    term2 = (
        np.add(
            np.maximum(
                np.subtract(np.add(X255, np.maximum(X234, X17)), X26),
                np.maximum(X221, X17)
            ),
            np.minimum(np.sqrt(X239), np.add(X179, X149))
        )
    )

    result = np.add(term1, term2)

    return pd.Series(result)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/353/console
# 1540_X06
# add(X245, add(X192, add(add(cos(X207), cos(X207)), add(add(X234, cos(add(mjin(X179, X245), add(add(add(X245, add(div(X245, X27), add(add(X193, cos(X207)), add(X157, X240)))), min(X179, add(X245, X234))), add(X245, min(X179, min(div(X245, X27), min(X179, min(X179, min(X179, add(X245, X234))))))))))), min(X179, min(min(add(X245, X234), add(X245, X234)), div(X240, X27)))))))
############################################################################
def create_gp_feature_1540_X06(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X207 = _df['hotelmarket_segment_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X27 =  _df['total_of_special_requests']
    X193 = _df['p_low']
    X157 = _df['binning_labels_adr_208_234']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    
    term1 = np.add(X245, np.add(X192, np.add(
        np.add(
            np.cos(X207),
            np.cos(X207)
        ),
        np.add(
            np.add(
                X234,
                np.cos(
                    np.add(
                        np.minimum(X179, X245),
                        np.add(
                            np.add(
                                np.add(
                                    X245,
                                    np.where(X27 != 0, np.divide(X245, X27), 0)  # 0割対策
                                ),
                                np.add(
                                    np.add(
                                        X193,
                                        np.cos(X207)
                                    ),
                                    np.add(X157, X240)
                                )
                            ),
                            np.where(X179 != 0, np.minimum(X179, np.add(X245, X234)), 0)  # 0割対策
                        )
                    )
                )
            ),
            np.minimum(
                X179,
                np.minimum(
                    np.add(X245, X234),
                    np.add(X245, X234)
                )
            )
        )
    )))

    term2 = np.where(X27 != 0, np.divide(X240, X27), 0)  # 0割対策

    result = np.add(term1, term2)

    return pd.Series(result)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/464/console
# 2349_X02
# add(add(add(add(add(min(max(max(X183, X17), X234), X2), sin(cos(max(X26, X184)))), min(inv(X47), min(max(sin(X33), X234), max(X27, X234)))), min(max(max(X17, X264), X264), X255)), max(add(cos(X27), X265), add(abs(X139), cos(max(min(inv(X47), X255), sin(X33)))))), min(max(X183, X234), min(max(X17, X264), X255)))
############################################################################
def create_gp_feature_2349_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X2 =   _df['lead_time']
    X17 =  _df['previous_cancellations']
    X26 =  _df['required_car_parking_spaces']
    X27 =  _df['total_of_special_requests']
    X33 =  _df['lead_time*arrival_date_week_number']
    X47 =  _df['hotelreserved_room_type_cat_combination_2']
    X139 = _df['binning_labels_lead_time_180_200']
    X183 = _df['relation_market_segment_and_deposit_type_and_customer_type']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.add(
        np.add(
            np.add(
                np.add(
                    np.minimum(
                        np.maximum(
                            np.maximum(X183, X17),
                            X234
                        ),
                        X2
                    ),
                    np.sin(np.cos(np.maximum(X26, X184))),
                ),
                np.minimum(
                    np.negative(X47),
                    np.minimum(
                        np.maximum(np.sin(X33), X234),
                        np.maximum(X27, X234)
                    )
                ),
            ),
            np.maximum(
                np.add(
                    np.cos(X27),
                    X265
                ),
                np.add(
                    np.abs(X139),
                    np.cos(
                        np.maximum(
                            np.minimum(
                                np.negative(X47),
                                X255
                            ),
                            np.sin(X33)
                        )
                    )
                )
            )
        ),
        np.minimum(
            np.maximum(X183, X234),
            np.minimum(
                np.maximum(X17, X264),
                X255
            )
        )
    )

    term1 = np.where(np.isinf(term1), 0, term1)
    term1 = np.where(np.isnan(term1), 0, term1)
    
    return pd.Series(term1)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/471/console
# 3992_X08
# add(add(sin(add(X234, max(X234, X189))), add(max(X234, min(X195, sin(min(min(add(X234, max(X234, X189)), sin(X195)), sin(X195))))), X265)), add(max(sub(add(max(min(add(X234, min(add(min(add(X234, max(X234, X189)), sin(X195)), max(X234, X112)), max(X234, X189))), sin(X195)), max(X234, abs(X17))), X265), X27), X139), X212))
############################################################################
def create_gp_feature_3992_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X17 =  _df['previous_cancellations']
    X27 =  _df['total_of_special_requests']
    X112 = _df['math_stays_in_weekend_nights_mean_groupby_yyyy-mm-week']
    X139 = _df['binning_labels_lead_time_180_200']
    X189 = _df['domain_is_deposit']
    X195 = _df['dev_cancel_singularity_classification']
    X212 = _df['hotelcustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.add(
        np.add(
            np.sin(np.add(X234, np.maximum(X234, X189))),
            np.add(
                np.maximum(X234, np.minimum(X195, np.sin(np.minimum(
                    np.minimum(
                        np.add(X234, np.maximum(X234, X189)),
                        np.sin(X195)
                    ),
                    np.sin(X195)
                )))),
                X265
            )
        ),
        np.add(
            np.maximum(
                np.subtract(
                    np.add(
                        np.maximum(
                            np.add(X234, np.minimum(
                                np.add(
                                    np.minimum(
                                        np.add(X234, np.maximum(X234, X189)),
                                        np.sin(X195)
                                    ),
                                    np.maximum(X234, X112)
                                ),
                                np.maximum(X234, X189)
                            )),
                            np.sin(X195)
                        ),
                        X265
                    ),
                    X27
                ),
                X139
            ),
            X212
        )
    )

    term1 = np.where(np.isinf(term1), 0, term1)
    term1 = np.where(np.isnan(term1), 0, term1)

    return pd.Series(term1)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/473/console
# 3173_X09
# add(add(add(add(X265, add(sin(max(max(inv(X192), X234), X182)), add(sin(add(max(X202, X192), add(X245, X240))), add(abs(X179), X262)))), add(sin(add(X245, X240)), max(X202, X192))), neg(X220)), max(cos(X137), add(X139, X234)))
############################################################################
def create_gp_feature_3137_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X137 = df['binning_labels_lead_time_140_160']
    X139 = df['binning_labels_lead_time_180_200']
    X179 = df['starndard_lead_time']
    X182 = df['relation_market_segment_and_deposite_type']
    X192 = df['domain_previous_booking_minus_canceled_and_notcanceled']
    X202 = df['customer_type_target_encoder']
    X220 = df['arrival_date_monthassigned_room_type_cat_combination_2_target_encoder']
    X234 = df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X262 = df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X265 = df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.add(
        np.add(
            np.add(
                np.add(
                    X265,
                    np.add(
                        np.sin(
                            np.maximum(
                                np.maximum(
                                    np.negative(X192),
                                    X234
                                ),
                                X182
                            )
                        ),
                        np.add(
                            np.sin(
                                np.add(
                                    np.maximum(X202, X192),
                                    np.add(X245, X240)
                                )
                            ),
                            np.add(abs(X179), X262)
                        )
                    )
                ),
                np.add(
                    np.sin(
                        np.add(X245, X240)
                    ),
                    np.maximum(X202, X192)
                )
            ),
            np.negative(X220)
        ),
        np.maximum(np.cos(X137), np.add(X139, X234))
    )

    term1 = np.where(np.isinf(term1), 0, term1)
    term1 = np.where(np.isnan(term1), 0, term1)

    return pd.Series(term1)



############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/478/console
# 613_X07
# add(sin(add(add(X192, add(X265, X234)), neg(X130))), add(sqrt(add(X192, add(sin(add(X234, X234)), neg(X130)))), add(max(X192, X139), add(add(X265, sqrt(add(neg(X138), sqrt(add(X192, add(X265, X234)))))), add(neg(X26), add(add(X265, sin(add(X192, add(X234, X234)))), sqrt(sqrt(add(X192, X240)))))))))
############################################################################
def create_gp_feature_613_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X130 = _df['binning_labels_lead_time_0_20']
    X138 = _df['binning_labels_lead_time_160_180']
    X139 = _df['binning_labels_lead_time_180_200']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']

    term1 = np.add(
        np.sin(
            np.add(
                np.add(
                    X192,
                    np.add(X265, X234)
                ),
                np.negative(X130)
            )
        ),
        np.add(
            np.sqrt(
                np.add(
                    X192,
                    np.add(
                        np.sin(
                            np.add(X234, X234)
                        ),
                        np.negative(X130)
                    )
                )
            ),
            np.add(
                np.maximum(X192, X139),
                np.add(
                    np.add(
                        X265,
                        np.sqrt(
                            np.add(
                                np.negative(X138),
                                np.sqrt(
                                    np.add(
                                        X192,
                                        np.add(
                                            X265,
                                            X234
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    np.add(
                        np.negative(X26),
                        np.add(
                            np.add(
                                X265,
                                np.sin(
                                    np.add(
                                        X192,
                                        np.add(
                                            X234,
                                            X234
                                        )
                                    )
                                )
                            ),
                            np.sqrt(
                                np.sqrt(
                                    np.add(
                                        X192,
                                        X240
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    term1 = np.where(np.isinf(term1), 0, term1)
    term1 = np.where(np.isnan(term1), 0, term1)

    return pd.Series(term1)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/496/console
# 2491_X04
# max(X183, sin(add(X192, sin(add(min(X139, sin(min(X195, min(X195, X234)))), sin(sin(add(neg(X26), add(min(X195, min(X195, sin(X234))), sin(add(X141, sin(X234))))))))))))
############################################################################
def create_gp_feature_2491_X04(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X183 = df['relation_market_segment_and_deposit_type_and_customer_type']
    X192 = df['domain_previous_booking_minus_canceled_and_notcanceled']
    X139 = df['binning_labels_lead_time_180_200']
    X195 = df['dev_cancel_singularity_classification']
    X234 = df['countrymarket_segment_cat_combination_2_target_encoder']
    X26 =  df['required_car_parking_spaces']
    X141 = df['binning_labels_lead_time_220_240']
    
    term1 = np.maximum(
        X183,
        np.sin(
            np.add(
                X192,
                np.sin(
                    np.add(
                        np.minimum(
                            X139,
                            np.sin(
                                np.minimum(
                                    X195,
                                    np.minimum(
                                        X195,
                                        X234
                                    )
                                )
                            )
                        ),
                        np.sin(
                            np.sin(
                                np.add(
                                    np.negative(X26),
                                    np.add(
                                        np.minimum(X195, np.minimum(X195, np.sin(X234))),
                                        np.sin(
                                            np.add(
                                                X141,
                                                np.sin(X234)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return pd.Series(term1)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/503/console
# 2294_X10
# sub(cos(X265), mul(mul(mul(mul(mul(max(X234, X192), cos(max(X26, X189))), cos(max(X234, max(X26, X189)))), cos(max(X26, X189))), cos(max(sin(max(sqrt(X27), max(sin(max(abs(X26), max(sqrt(X27), X234))), sqrt(sub(X130, X134))))), max(X130, X182)))), cos(max(sin(X19), mul(div(X182, X195), max(X234, X192))))))
############################################################################
def create_gp_feature_2294_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X26 =  _df['required_car_parking_spaces']
    X189 = _df['domain_is_deposit']
    X27 =  _df['total_of_special_requests']
    X130 = _df['binning_labels_lead_time_0_20']
    X134 = _df['binning_labels_lead_time_80_100']
    X182 = _df['relation_market_segment_and_deposite_type']
    X195 = _df['dev_cancel_singularity_classification']
    X19 =  _df['reserved_room_type']

    term1 = np.cos(X265)

    term2 = np.multiply(
        np.multiply(
            np.multiply(
                np.multiply(
                    np.multiply(
                        np.maximum(X234, X192),
                        np.cos(np.maximum(X26, X189))
                    ),
                    np.cos(np.maximum(X234, np.maximum(X26, X189)))
                ),
                np.cos(np.maximum(X26, X189))
            ),
            np.cos(np.maximum(X26, X189))
        ),
        np.cos(
            np.maximum(
                np.sin(
                    np.maximum(
                        np.sqrt(X27),
                        np.maximum(
                            np.sin(
                                np.maximum(
                                    np.abs(X26),
                                    np.maximum(
                                        np.sqrt(X27),
                                        X234
                                    )
                                )
                            ),
                            np.sqrt(np.subtract(X130, X134))
                        )
                    )
                ),
                np.maximum(X130, X182)
            )
        )
    )

    term3 = np.cos(
        np.maximum(
            np.sin(X19),
            np.multiply(
                np.divide(X182, X195),
                np.maximum(X234, X192)
            )
        )
    )

    result = np.subtract(term1, np.subtract(term2, term3))
    result = np.where(np.isinf(result), 0, result)
    result = np.where(np.isnan(result), 0, result)

    return pd.Series(result)


############################################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/504/console
# 4535_X03
# cos(sin(mul(sqrt(max(X265, X192)), add(X234, mul(mul(mul(mul(sqrt(max(mul(mul(sqrt(add(sqrt(max(add(sqrt(max(sqrt(X178), X192)), X192), max(cos(min(sqrt(X178), X26)), cos(X26)))), X192)), sqrt(cos(X26))), X179), X192)), sin(add(X234, add(X234, max(max(X265, X192), X234))))), cos(min(max(sqrt(X178), X192), X26))), cos(X26)), cos(X26))))))
############################################################################
def create_gp_feature_4535_X03(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X265 = df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = df['domain_previous_booking_minus_canceled_and_notcanceled']
    X234 = df['countrymarket_segment_cat_combination_2_target_encoder']
    X26 =  df['required_car_parking_spaces']
    X179 = df['starndard_lead_time']
    X178 = df['dev_value_counts_adr']

    term14 = np.minimum(
        np.sqrt(X178),
        X26
    )

    term11 = np.add(
        np.maximum(
            np.cos(term14),
            np.cos(X26)
        ),
        X192
    )

    term13 = np.add(
        np.sqrt(
            np.maximum(
                np.add(
                    np.sqrt(
                        np.maximum(
                            np.sqrt(X178),
                            X192
                        )
                    ),
                    X192
                ),
                X192
            )
        ),
        X192
    )

    term10 = np.add(
        np.sqrt(term13),
        X192
    )

    term6 = np.maximum(
        X265,
        X192
    )

    term4 = np.sqrt(term6)

    term12 = np.add(
        X234,
        np.maximum(
            np.maximum(
                X265,
                X192
            ),
            X234
        )
    )

    term9 = np.sin(term12)

    term8 = np.multiply(
        np.multiply(
            term4,
            X192
        ),
        np.sqrt(term11)
    )

    term7 = np.multiply(
        np.multiply(
            X234,
            term8
        ),
        term9
    )

    term5 = np.add(
        X234,
        term7
    )

    term3 = np.multiply(
        term4,
        term5
    )

    term2 = np.sin(term3)

    term1 = np.cos(term2)

    return pd.Series(term1)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/506/console
# 3664_X02
# sub(add(X245, add(min(X179, X240), X192)), cos(add(X179, add(add(add(X193, X192), min(X245, sub(min(X179, X240), cos(add(add(add(X245, min(X179, min(X234, X234))), X192), add(X245, min(X179, X234))))))), X234))))
############################################################################countrymarket_segment_cat_combination_2_target_enccountrymarket_segment_cat_combination_2_target_enccountrymarket_segment_cat_combination_2_target_enccountrymarket_segment_cat_combination_2_target_enccountrymarket_segment_cat_combination_2_target_encodercountrymarket_segment_cat_combination_2_target_encode[
def create_gp_feature_3364_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X245 = _df['distribution_channelassigned_room_type_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X193 = _df['p_low']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']

    term14 = np.add(
        X245,
        np.minimum(
            X179,
            X234
        )
    )
    
    term13 = np.minimum(
        X179,
        np.minimum(
            X234,
            X234
        )
    )
    
    term12 = np.add(
        np.add(
            np.add(
                X245,
                term13
            ),
            X192
        ),
        term14
    )
    
    term11 = np.cos(term12)
    
    term10 = np.subtract(
        np.minimum(
            X179,
            X240
        ),
        term11
    )
    
    term8 = np.minimum(
        X245,
        term10
    )
    
    term7 = np.add(
        X193,
        X192
    )
    
    term6 = np.add(
        np.add(
            term7,
            term8
        ),
        X234
    )
    
    term5 = np.add(
        X179,
        term6
    )
    
    term3 = np.cos(term5)
    
    term4 = np.add(
        np.minimum(
            X179,
            X240
        ),
        X192
    )
    
    term2 = np.add(
        X245,
        term4
    )
    
    term1 = np.subtract(
        term2,
        term3
    )
    
    return pd.Series(term1)



############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/521/console
# 1372_X04
# add(add(div(X214, X26), add(mul(X255, tan(X240)), add(sqrt(X234), X265))), min(mul(tan(tan(X252)), mul(add(mul(X99, add(sqrt(X234), X265)), add(tan(tan(X252)), add(sqrt(X234), mul(X169, X17)))), mul(add(sqrt(X234), X265), mul(X255, tan(X240))))), X195))
############################################################
def create_gp_feature_1372_X04(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X214 = _df['hotelfeature_seasons_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X99 =  _df['assigned_room_typecustomer_type_cat_combination_2']
    X169 = _df['dev_value_counts_stays_in_week_nights']
    X17 =  _df['previous_cancellations']
    X195 = _df['dev_cancel_singularity_classification']

    term1 = np.where(X26 != 0, np.divide(X214, X26), 0)
    term2 = np.multiply(X255, np.tan(X240))
    term3 = np.sqrt(X234)
    term4 = np.add(term2, term3)
    term5 = np.add(term1, term4)
    term6 = np.tan(np.tan(X252))
    term7 = np.multiply(X99, term3)
    term8 = np.multiply(X169, X17)
    term9 = np.add(term3, term8)
    term10 = np.add(term7, term9)
    term11 = np.multiply(term6, term10)
    term12 = np.multiply(term3, term2)
    term13 = np.multiply(term11, term12)
    term14 = np.minimum(term13, X195)
    term15 = np.add(term5, term14)

    return pd.Series(term15)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/526/console
# 59_X05
# add(cos(add(add(X262, add(cos(add(X219, sin(sin(add(X219, add(X219, sin(min(X95, X193)))))))), add(sqrt(add(sqrt(min(X264, X264)), X264)), add(X265, add(sin(add(add(X234, X234), abs(X17))), X179))))), add(sqrt(sqrt(sqrt(sqrt(mul(min(X95, X193), X240))))), sin(add(abs(X17), X264))))), sin(min(min(add(add(min(X161, sqrt(add(X219, sin(add(X219, add(sqrt(X234), sqrt(add(X17, X264)))))))), X264), X234), X25), X25)))
############################################################
def create_gp_feature_59_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X262 = _df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X219 = _df['arrival_date_monthreserved_room_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X95 =  _df['reserved_room_typecustomer_type_cat_combination_2']
    X193 = _df['p_low']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X161 = _df['binning_labels_adr_312_338']
    X25 =  _df['adr']
    X179 = _df['starndard_lead_time']

    term1 = np.sin(np.sin(np.add(X219, np.add(np.sqrt(X234), np.sqrt(np.add(X17, X264))))))
    term2 = np.cos(np.add(X219, term1))
    term3 = np.sqrt(np.add(np.sqrt(np.minimum(X264, X264)), X264))
    term4 = np.sin(np.add(np.add(X234, X234), np.abs(X17)))
    term5 = np.add(X265, np.add(term4, X179))
    term6 = np.add(term3, term5)
    term7 = np.add(X262, np.add(term2, term6))
    term8 = np.cos(term7)
    term9 = np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.multiply(np.minimum(X95, X193), X240)))))
    term10 = np.sin(np.add(np.abs(X17), X264))
    term11 = np.add(term9, term10)
    term12 = np.add(term8, term11)
    term13 = np.minimum(np.add(np.add(np.minimum(X161, np.sqrt(np.add(X219, term1))), X264), X234), X25)
    term14 = np.sin(np.minimum(term13, X25))
    term15 = np.add(term12, term14)

    return pd.Series(term15)

############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/551/console
# 580_X09
# sub(max(mul(add(X255, X245), sin(X239)), X182), mul(neg(sin(add(mul(add(X234, min(X239, X179)), sin(add(add(X255, X245), X255))), mul(add(X255, X245), min(min(sin(X234), X179), X98))))), X179))
############################################################
def create_gp_feature_580_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X182 = _df['relation_market_segment_and_deposite_type']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X98 =  _df['assigned_room_typedeposit_type_cat_combination_2']

    term0 = np.add(X255, X245)
    term1 = np.sin(X239)
    term2 = np.multiply(term0, term1)
    term3 = np.maximum(term2, X182)
    term4 = np.minimum(np.sin(X234), X179)
    term5 = np.minimum(term4, X98)
    term6 = np.add(X255, X245)
    term7 = np.add(term6, X255)
    term8 = np.sin(term7)
    term9 = np.add(X234, term5)
    term10 = np.multiply(term9, term8)
    term11 = np.sin(np.add(term10, term6))
    term12 = np.negative(term11)
    term13 = np.multiply(term12, X179)
    term14 = np.subtract(term3, term13)

    return pd.Series(term14)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/593/console
# 4974_X08
# add(sin(mul(sin(mul(sqrt(sqrt(sqrt(sqrt(min(min(X240, X180), X193))))), add(add(max(X234, abs(X17)), min(X234, X195)), min(X234, abs(X195))))), add(add(max(X234, sqrt(max(sin(X17), min(min(sqrt(sqrt(sqrt(sqrt(max(X234, sqrt(min(min(mul(sqrt(sqrt(inv(X195))), min(X240, X180)), X193), X193))))))), X193), min(X234, inv(sqrt(X240))))))), min(X234, sub(X234, X26))), sub(X234, X26)))), sqrt(X265))
############################################################
def create_gp_feature_4974_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X180 = _df['starndard_adr']
    X193 = _df['p_low']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X195 = _df['dev_cancel_singularity_classification']
    X26 =  _df['required_car_parking_spaces']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term0 = np.minimum(X240, X180)
    term1 = np.minimum(term0, X193)
    term2 = np.sqrt(term1)
    term3 = np.sqrt(term2)
    term4 = np.sqrt(term3)
    term5 = np.sqrt(term4)
    term6 = np.abs(X17)
    term7 = np.maximum(X234, term6)
    term8 = np.minimum(X234, X195)
    term9 = np.add(term7, term8)
    term10 = np.abs(X195)
    term11 = np.minimum(X234, term10)
    term12 = np.add(term9, term11)
    term13 = np.sqrt(term7)
    term14 = np.minimum(term13, X193)
    term15 = np.sqrt(term14)
    term16 = np.sin(X17)
    term17 = np.sqrt(term15)
    term18 = np.maximum(term16, term17)
    term19 = np.sqrt(term18)
    term20 = np.maximum(term19, X193)
    term21 = np.sqrt(term20)
    term22 = np.maximum(X234, term21)
    term23 = 1/(np.sqrt(X240))
    term24 = np.minimum(X234, term23)
    term25 = np.add(term22, term24)
    term26 = np.maximum(X234, term25)
    term27 = np.subtract(X234, X26)
    term28 = np.minimum(X234, term27)
    term29 = np.add(term26, term28)
    term30 = np.subtract(X234, X26)
    term31 = np.add(term29, term30)
    term32 = np.multiply(term12, term31)
    term33 = np.sin(term32)
    term34 = np.multiply(term5, term33)
    term35 = np.sin(term34)
    term36 = np.multiply(term35, term12)
    term37 = np.sin(term36)
    term38 = np.add(term37, term12)
    term39 = np.sin(term38)
    term40 = np.sqrt(X265)
    term41 = np.add(term39, term40)

    return pd.Series(term41)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/605/console
# 3001_X02
# sin(sin(sub(sub(sub(X190, add(add(sin(sin(sin(add(X234, X239)))), X265), X179)), add(X192, sin(add(sin(sin(add(X234, X239))), X265)))), sin(sin(cos(mul(sqrt(X26), add(add(sin(sub(X190, sub(X190, add(add(sqrt(X26), X265), X179)))), X265), X179))))))))
############################################################
def create_gp_feature_3001_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X190 = _df['domain_is_day_trip']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X26 =  _df['required_car_parking_spaces']

    term0 = np.add(X234, X239)
    term1 = np.sin(term0)
    term2 = np.sin(term1)
    term3 = np.sin(term2)
    term4 = np.add(term3, X265)
    term5 = np.add(term4, X179)
    term6 = np.subtract(X190, term5)
    term7 = np.add(term3, X265)
    term8 = np.sin(term7)
    term9 = np.add(X192, term8)
    term10 = np.subtract(term6, term9)
    term11 = np.sqrt(X26)
    term12 = np.add(term4, X265)
    term13 = np.add(term12, X179)
    term14 = np.multiply(term11, term13)
    term15 = np.cos(term14)
    term16 = np.sin(term15)
    term17 = np.sin(term16)
    term18 = np.subtract(term10, term17)
    term19 = np.sin(term18)
    term20 = np.sin(term19)

    return pd.Series(term20)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/608/console
# 3562_X05
# add(add(max(X17, X265), max(X264, min(min(min(div(inv(X17), X137), min(min(min(div(min(X26, X61), X137), add(min(X195, add(X255, X255)), X234)), div(min(X234, X22), min(X26, X61))), div(min(X234, X22), min(X195, X255)))), div(min(min(X17, min(add(X255, X234), X255)), X265), min(X26, X61))), div(min(X234, X22), min(X195, X255))))), min(min(div(inv(X17), min(X26, X61)), add(X255, X255)), X255))
############################################################
def create_gp_feature_3562_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X17 =  _df['previous_cancellations']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X137 = _df['binning_labels_lead_time_140_160']
    X26 =  _df['required_car_parking_spaces']
    X61 =  _df['arrival_date_monthyyyy-mm-week_cat_combination_2']
    X255 = _df['distribution_channelassigned_room_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X22 = _df['agent']

    # 対象箇所の修正
    term0 = np.maximum(X17, X265)
    term1 = np.minimum(X26, X61)
    term2 = np.where(X17 != 0, 1 / X17, 0)
    term3 = np.where(X137 != 0, term2 / X137, 0)
    term4 = np.minimum(term3, term1)
    term5 = np.add(X255, X234)
    term6 = np.minimum(term5, X255)
    term7 = np.minimum(X17, term6)
    term8 = np.minimum(term7, X265)
    term9 = np.minimum(term8, term1)
    term10 = np.where(term9 != 0, 1 / term9, 0)
    term11 = np.minimum(X234, X22)
    term12 = np.where(term11 != 0, term10 / term11, 0)
    term13 = np.minimum(term4, term12)
    term14 = np.where(X137 != 0, term1 / X137, 0)
    term15 = np.add(term6, X234)
    term16 = np.minimum(term14, term15)
    term17 = np.where(X137 != 0, term16 / X137, 0)
    term18 = np.minimum(term17, term16)
    term19 = np.minimum(term13, term18)
    term20 = np.minimum(X234, X22)
    term21 = np.minimum(X195, X255)
    term22 = np.where(term21 != 0, term19 / term21, 0)
    term23 = np.minimum(X264, term22)
    term24 = np.add(term0, term23)
    term25 = np.where(term1 != 0, term2 / term1, 0)
    term26 = np.add(X255, X255)
    term27 = np.minimum(term25, term26)
    term28 = np.minimum(term27, X255)
    term29 = np.add(term24, term28)

    # 対象箇所の修正
    result = np.sin(np.sin(np.where(~np.isnan(term29), term29, 0)))

    return pd.Series(result, name='created_feature')
    
    
############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/611/console
# 792_X08
# max(mul(X266, X266), min(add(mul(add(max(X192, sub(max(X192, min(X179, X32)), inv(inv(max(X139, X27))))), X139), X234), max(X192, min(sub(mul(max(X192, min(X179, X32)), max(X192, min(X179, X32))), X26), max(max(max(X192, sub(max(X192, min(X22, max(X192, X234))), mul(X266, X266))), X234), X234)))), sub(max(X192, min(X179, X32)), X26)))
############################################################
def create_gp_feature_792_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X17 =  _df['previous_cancellations']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X137 = _df['binning_labels_lead_time_140_160']
    X26 =  _df['required_car_parking_spaces']
    X61 =  _df['arrival_date_monthyyyy-mm-week_cat_combination_2']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X32 =  _df['lead_time*adr']
    X139 = _df['binning_labels_lead_time_180_200']
    X27 =  _df['total_of_special_requests']
    X266 = _df['deposit_typefeature_seasons_cat_combination_2_target_encoder']

    term0 = np.maximum(X192, X265)
    term1 = np.maximum(X192, np.minimum(X179, X32))
    term2 = np.subtract(term0, term1)
    term3 = np.maximum(X139, X27)
    term4 = 1/term3
    term5 = 1/term4
    term6 = np.subtract(term0, term5)
    term7 = np.maximum(X192, term6)
    term8 = np.add(term7, X139)
    term9 = np.multiply(term8, term1)
    term10 = np.add(term9, X234)
    term11 = np.minimum(term10, term0)
    term12 = np.subtract(term1, X26)
    term13 = np.minimum(term12, term0)
    term14 = np.maximum(term13, term0)
    term15 = np.maximum(term14, X234)
    term16 = np.maximum(term15, X234)
    term17 = np.subtract(term1, X26)
    term18 = np.minimum(term11, term16)
    term19 = np.multiply(X266, X266)
    term20 = np.maximum(term19, term18)

    return pd.Series(term20)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/628/console
# 3099_X06
# add(cos(add(add(min(sin(sin(min(X234, X255))), cos(div(abs(X3), sin(X26)))), min(X265, X41)), add(min(X234, mul(X179, add(mul(X179, cos(log(X102))), sin(min(X234, X255))))), X265))), min(X265, X41))
############################################################
def create_gp_feature_3099_X06(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X3 =   _df['arrival_date_year']
    X26 =  _df['required_car_parking_spaces']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X41 =  _df['adr*arrival_date_week_number*arrival_date_day_of_month']
    X179 = _df['starndard_lead_time']
    X102 = _df['deposit_typecustomer_type_cat_combination_2']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']

    term0 = np.minimum(X234, X255)
    term1 = np.sin(term0)
    term2 = np.sin(term1)
    term3 = np.sin(term2)
    term4 = np.abs(X3)
    term5 = np.sin(X26)
    term6 = np.where(term5 != 0, term4 / term5, 0)
    term7 = np.cos(term6)
    term8 = np.add(term3, term7)
    term9 = np.minimum(X265, X41)
    term10 = np.add(term8, term9)
    term11 = np.multiply(X179, np.log(X102))
    term12 = np.cos(term11)
    term13 = np.multiply(X179, term12)
    term14 = np.add(term13, term2)
    term15 = np.multiply(X179, term14)
    term16 = np.minimum(X234, term15)
    term17 = np.add(term16, X265)
    term18 = np.minimum(X265, X41)
    term19 = np.add(term10, term17)
    term20 = np.cos(term19)
    term21 = np.add(term20, term18)

    return pd.Series(term21)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/641/console
# 4111_X10
# sin(sub(min(sqrt(X265), add(sub(min(min(X179, X234), cos(tan(sin(X192)))), sub(cos(sqrt(X265)), sqrt(X265))), min(X179, min(X179, X234)))), sub(min(X179, sqrt(X265)), sub(min(min(X179, sqrt(X265)), add(X234, X189)), sub(abs(sqrt(sub(cos(sin(X192)), sqrt(X265)))), sin(sub(sub(min(cos(sqrt(X265)), min(X179, sqrt(X265))), sub(cos(sin(X192)), sqrt(X265))), sub(X26, sub(min(min(X179, sqrt(X265)), min(X179, add(X234, X189))), sub(abs(sub(X236, X251)), sub(sin(X192), min(X27, sqrt(X265)))))))))))))
############################################################
def create_gp_feature_4111_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X3 =   _df['arrival_date_year']
    X26 =  _df['required_car_parking_spaces']
    X41 =  _df['adr*arrival_date_week_number*arrival_date_day_of_month']
    X189 = _df['domain_is_deposit']
    X236 = _df['countryreserved_room_type_cat_combination_2_target_encoder']
    X251 = _df['distribution_channeldeposit_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X102 =  _df['deposit_typecustomer_type_cat_combination_2']
    
    term0 = np.sqrt(X265)
    term1 = np.minimum(X179, X234)
    term2 = np.sin(term1)
    term3 = np.sin(term2)
    term4 = np.sin(term3)
    term5 = np.abs(X3)
    term6 = np.sin(X26)
    term7 = np.where(term6 != 0, term5 / term6, 0)
    term8 = np.cos(term7)
    term9 = np.add(term4, term8)
    term10 = np.minimum(X265, X41)
    term11 = np.add(term9, term10)
    term12 = np.multiply(X179, np.log(X102))
    term13 = np.cos(term12)
    term14 = np.multiply(X179, term13)
    term15 = np.add(term14, term2)
    term16 = np.multiply(X179, term15)
    term17 = np.minimum(X234, term16)
    term18 = np.add(term17, X265)
    term19 = np.minimum(X265, X41)
    term20 = np.add(term11, term18)
    term21 = np.cos(term20)
    term22 = np.add(term21, term19)

    return pd.Series(np.sin(term22))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/646/console
# 4409_X09
# add(max(tan(X221), max(min(X216, X41), add(min(X179, add(X234, min(X216, mul(X255, X234)))), mul(X255, X234)))), min(min(X93, sin(X234)), min(min(X255, min(X255, X41)), min(X195, add(X255, X234)))))
############################################################
def create_gp_feature_4409_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X221 = _df['arrival_date_monthdeposit_type_cat_combination_2_target_encoder']
    X216 = _df['arrival_date_monthcountry_cat_combination_2_target_encoder']
    X41 =  _df['adr*arrival_date_week_number*arrival_date_day_of_month']
    X179 = _df['starndard_lead_time']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X93 =  _df['reserved_room_typeassigned_room_type_cat_combination_2']
    X195 = _df['dev_cancel_singularity_classification']
    X189 = _df['domain_is_deposit']

    term0 = np.tan(X221)
    term1 = np.minimum(X216, X41)
    term2 = np.maximum(X179, term1)
    term3 = np.multiply(X255, X234)
    term4 = np.minimum(X216, term3)
    term5 = np.add(X234, term4)
    term6 = np.minimum(X179, term5)
    term7 = np.add(term6, X234)
    term8 = np.maximum(term2, term7)
    term9 = np.minimum(X93, np.sin(X234))
    term10 = np.minimum(X255, X41)
    term11 = np.minimum(X255, term10)
    term12 = np.add(X255, X234)
    term13 = np.minimum(X195, term12)
    term14 = np.minimum(term11, term13)
    term15 = np.minimum(term9, term14)
    term16 = np.add(term8, term15)

    return pd.Series(np.sin(term16))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/650/console
# 4831_X08
# mul(sqrt(add(X253, sin(sin(sin(add(X234, log(add(X195, X265)))))))), add(X265, sin(sin(add(max(sin(add(X255, add(add(X234, sin(sin(add(add(X255, X234), log(add(X255, X234)))))), mul(X26, X234)))), log(mul(X26, X91))), sin(add(add(sin(add(add(X234, sin(sin(add(sqrt(add(X253, sin(sin(sin(add(add(X234, X253), log(add(X265, X234)))))))), log(add(X255, X234)))))), mul(X26, log(X251)))), log(mul(X26, X91))), log(add(X265, X216)))))))))
############################################################
def create_gp_feature_4831_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X253 = _df['distribution_channelyyyy-mm-week_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X216 = _df['arrival_date_monthcountry_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X221 = _df['arrival_date_monthdeposit_type_cat_combination_2_target_encoder']
    X93 =  _df['reserved_room_typeassigned_room_type_cat_combination_2']
    X41 =  _df['adr*arrival_date_week_number*arrival_date_day_of_month']
    X189 = _df['domain_is_deposit']

    term0 = np.add(X234, np.log(np.add(X195, X265)))
    term1 = np.sin(term0)
    term2 = np.sin(term1)
    term3 = np.sin(term2)
    term4 = np.add(X253, term3)
    term5 = np.sqrt(np.where(term4 >= 0, term4, 0))
    term6 = np.minimum(X179, np.add(X234, np.minimum(X216, np.multiply(X255, X234))))
    term7 = np.add(term6, np.multiply(X255, X234))
    term8 = np.add(term5, term7)
    term9 = np.sin(np.sin(np.add(np.maximum(np.sin(np.add(X255, term8)), np.log(np.multiply(X26, X221))), np.sin(np.add(term8, np.log(np.add(X265, X216)))))))
    term10 = np.minimum(X93, np.sin(X234))
    term11 = np.minimum(X255, term10)
    term12 = np.minimum(X255, term11)
    term13 = np.minimum(X195, np.add(X255, X234))
    term14 = np.minimum(term12, term13)
    term15 = np.add(term9, term14)

    return pd.Series(np.multiply(term5, term15))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/668/console
# 3883_X06
# min(sub(log(sin(X195)), log(max(max(max(sqrt(mul(mul(neg(X234), sin(X176)), cos(sub(X202, log(inv(X265)))))), min(sqrt(X17), max(cos(X216), min(log(inv(X265)), X234)))), mul(neg(X234), sin(X176))), sin(X265)))), cos(X216))
############################################################
def create_gp_feature_3883_X06(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X195 = _df['dev_cancel_singularity_classification']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X176 = _df['dev_value_counts_agent']
    X202 = _df['customer_type_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X216 = _df['arrival_date_monthcountry_cat_combination_2_target_encoder']

    term1 = np.sin(X195)
    term2 = np.maximum(np.maximum(np.sqrt(np.multiply(np.negative(X234), np.sin(X176))), np.minimum(np.sqrt(X17), np.maximum(np.cos(X216), np.minimum(np.log(np.where(X265!=0, 1/X265, 0)), X234)))), np.multiply(np.negative(X234), np.sin(X176)))
    term3 = np.sin(X265)
    term4 = np.cos(X216)
    term5 = np.log(np.where(term1>0, term1, 1e-10))
    term6 = np.log(np.where(term2>0, term2, 1e-10))
    term7 = np.minimum(term5, term6)
    term8 = np.minimum(term7, term3)

    return pd.Series(np.where(term4!=0, term8/term4, 0))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/676/console
# 2349_X02
# add(add(add(add(add(min(max(max(X183, X17), X234), X2), sin(cos(max(X26, X184)))), min(inv(X47), min(max(sin(X33), X234), max(X27, X234)))), min(max(max(X17, X264), X264), X255)), max(add(cos(X27), X265), add(abs(X139), cos(max(min(inv(X47), X255), sin(X33)))))), min(max(X183, X234), min(max(X17, X264), X255)))
############################################################
def create_gp_feature_2349_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X183 = df['relation_market_segment_and_deposit_type_and_customer_type']
    X17 =  df['previous_cancellations']
    X234 = df['countrymarket_segment_cat_combination_2_target_encoder']
    X2 =   df['lead_time']
    X26 =  df['required_car_parking_spaces']
    X184 = df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X47 =  df['hotelreserved_room_type_cat_combination_2']
    X33 =  df['lead_time*arrival_date_week_number']
    X27 =  df['total_of_special_requests']
    X264 = df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X255 = df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X265 = df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X139 = df['binning_labels_lead_time_180_200']

    term1 = np.maximum(X183, X17)
    term2 = np.maximum(term1, X234)
    term3 = np.sin(np.cos(np.maximum(X26, X184)))
    term4 = np.where(X47!=0, 1/X47, 0)
    term5 = np.sin(X33)
    term6 = np.maximum(term5, X234)
    term7 = np.maximum(X27, X234)
    term8 = np.minimum(term4, term6)
    term9 = np.minimum(term8, term7)
    term10 = np.maximum(X17, X264)
    term11 = np.maximum(term10, X264)
    term12 = np.cos(X27)
    term13 = np.abs(X139)
    term14 = np.cos(np.maximum(term4, term5))
    term15 = np.add(term12, X265)
    term16 = np.add(term13, term14)
    term17 = np.maximum(term15, term16)
    term18 = np.maximum(X183, X234)
    term19 = np.maximum(X17, X264)
    term20 = np.minimum(term18, term19)
    term21 = np.add(term2, term3)
    term22 = np.add(term21, term9)
    term23 = np.add(term22, term11)
    term24 = np.add(term23, term17)

    return pd.Series(np.minimum(term24, term20))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/680/console
# 160_X08
# add(sin(sin(add(add(add(log(add(sin(X234), X265)), mul(add(sin(div(X234, X265)), min(log(inv(X27)), min(min(neg(div(X27, X42)), X265), add(sin(X144), div(add(sin(X234), X265), X265))))), X179)), neg(X26)), div(div(mul(X240, X179), X265), X42)))), add(sin(sqrt(X268)), mul(X240, X179)))
############################################################
def create_gp_feature_160_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X42 =  _df['hotelarrival_date_month_cat_combination_2']
    X144 = _df['binning_labels_lead_time_280_300']
    X26 =  _df['required_car_parking_spaces']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X268 = _df['customer_typefeature_seasons_cat_combination_2_target_encoder']

    term1 = np.sin(X234)
    term2 = np.add(term1, X265)
    term3 = np.divide(X234, X265)
    term4 = np.where(X27!=0, 1/X27, 0)
    term5 = np.divide(term4, X42)
    term6 = np.negative(term5)
    term7 = np.minimum(term6, X265)
    term8 = np.sin(X144)
    term9 = np.add(term8, term3)
    term10 = np.minimum(term7, term9)
    term11 = np.log(np.where(term4>0, term4, 1e-10))
    term12 = np.minimum(term11, term10)
    term13 = np.sin(term3)
    term14 = np.add(term13, term12)
    term15 = np.multiply(term14, X179)
    term16 = np.log(np.where(term2>0, term2, 1e-10))
    term17 = np.add(term16, term15)
    term18 = np.negative(X26)
    term19 = np.add(term17, term18)
    term20 = np.sin(term19)
    term21 = np.sin(term20)
    term22 = np.sqrt(X268)
    term23 = np.sin(term22)
    term24 = np.multiply(X240, X179)
    term25 = np.add(term23, term24)
    term26 = np.add(term21, term25)

    return pd.Series(term26)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/692/console
# 1187_X03
# add(add(X157, add(max(min(X118, min(X118, min(X118, add(add(add(X234, add(abs(X180), add(add(add(add(add(add(add(X234, neg(X26)), neg(cos(X195))), X192), max(mul(X195, X216), X133)), neg(cos(X195))), X265), sin(X265)))), neg(cos(X195))), add(add(add(X234, neg(X26)), max(X192, X234)), add(log(sqrt(log(max(add(X234, neg(cos(X195))), log(X178))))), log(max(add(max(X192, X234), log(sqrt(log(max(sin(X265), log(X178)))))), X234)))))))), X234), max(sin(X265), X192))), mul(neg(X26), X267))
############################################################
def create_gp_feature_1187_X03(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X157 = _df['binning_labels_adr_208_234']
    X118 = _df['math_adults_mean_groupby_yyyy-mm-week']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X180 = _df['starndard_adr']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X195 = _df['dev_cancel_singularity_classification']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X216 = _df['arrival_date_monthcountry_cat_combination_2_target_encoder']
    X133 = _df['binning_labels_lead_time_60_80']
    X178 = _df['dev_value_counts_adr']
    X42 =  _df['hotelarrival_date_month_cat_combination_2']
    X144 = _df['binning_labels_lead_time_280_300']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X267 = _df['customer_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.sin(X234)
    term2 = np.add(term1, X265)
    term3 = np.negative(np.cos(X195))
    term4 = np.add(term2, term3)
    term5 = np.log(np.where(term4>0, term4, 1e-10))
    term6 = np.multiply(X195, X216)
    term7 = np.maximum(term6, X133)
    term8 = np.negative(np.cos(X195))
    term9 = np.add(term5, term7)
    term10 = np.add(term9, term8)
    term11 = np.add(term10, X265)
    term12 = np.sin(X265)
    term13 = np.add(term11, term12)
    term14 = np.minimum(X118, term13)
    term15 = np.minimum(X118, term14)
    term16 = np.minimum(X118, term15)
    term17 = np.add(term2, term16)
    term18 = np.log(np.where(term17>0, term17, 1e-10))
    term19 = np.add(term5, term18)
    term20 = np.sin(term19)
    term21 = np.sin(term20)
    term22 = np.add(X157, term21)
    term23 = np.sin(X265)
    term24 = np.maximum(term23, X192)
    term25 = np.multiply(np.negative(X26), X267)
    term26 = np.add(term22, term24)

    return pd.Series(np.add(term26, term25))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/697/console
# 931_X09
# add(min(min(tan(X195), inv(sqrt(sqrt(mul(min(min(X234, X32), X193), X262))))), add(add(X245, X239), sqrt(min(sqrt(mul(min(min(min(X234, X240), X193), X193), X262)), X239)))), add(add(X262, abs(add(add(neg(sqrt(X27)), X239), add(min(sqrt(sqrt(inv(sqrt(mul(min(X234, X240), X262))))), add(add(X245, X239), sqrt(min(X234, X32)))), add(add(X245, mul(X193, X262)), sqrt(min(X234, X32))))))), X245))
############################################################
def create_gp_feature_931_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X195 = _df['dev_cancel_singularity_classification']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X32 =  _df['lead_time*adr']
    X193 = _df['p_low']
    X262 = _df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X267 = _df['customer_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X180 = _df['starndard_adr']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.tan(X195)
    term2 = np.minimum(X234, X32)
    term3 = np.minimum(term2, X193)
    term4 = np.multiply(term3, X262)
    term5 = np.sqrt(np.where(term4>0, term4, 1e-10))
    term6 = np.sqrt(term5)
    term7 = np.where(term6!=0, 1/term6, 0)
    term8 = np.minimum(term1, term7)
    term9 = np.abs(X180)
    term10 = np.add(X234, term9)
    term11 = np.add(term10, X265)
    term12 = np.sin(term11)
    term13 = np.add(X245, X239)
    term14 = np.sqrt(np.where(term3>0, term3, 1e-10))
    term15 = np.minimum(term14, X239)
    term16 = np.add(term13, term15)
    term17 = np.add(term8, term16)
    term18 = np.sin(X265)
    term19 = np.add(X262, term18)
    term20 = np.negative(np.sqrt(X27))
    term21 = np.add(term20, X239)
    term22 = np.add(term19, term21)
    term23 = np.abs(term22)
    term24 = np.add(X262, term23)
    term25 = np.add(X245, term24)

    return pd.Series(np.add(term17, term25))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/698/console
# 2179_X05
# add(add(add(add(sin(add(mul(mul(X179, add(add(add(X214, X239), div(max(X157, X240), X27)), X239)), max(X245, X192)), X245)), mul(X179, add(div(X214, X27), X239))), max(X245, X192)), add(div(X214, X27), X239)), add(max(X245, X192), max(X157, X240))) 
############################################################
def create_gp_feature_2179_X05(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X179 = _df['starndard_lead_time']
    X214 = _df['hotelfeature_seasons_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X157 = _df['binning_labels_adr_208_234']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X262 = _df['assigned_room_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.multiply(X179, X214)
    term2 = np.add(X214, X239)
    term3 = np.divide(np.maximum(X157, X240), X27)
    term4 = np.add(term2, term3)
    term5 = np.multiply(X179, term4)

    # 対策: 分母が0の場合、0にする
    term6 = np.sin(np.add(term5, X245))
    term7 = np.add(X262, np.abs(np.add(np.negative(np.sqrt(X27)), X239)))
    term8 = np.add(term7, np.add(np.minimum(np.sqrt(np.sqrt(np.where(X214 != 0, 1 / X214, 0))), np.add(X245, X239)), np.add(X245, np.multiply(X192, X262))))
    term9 = np.add(term6, np.maximum(X245, X192))
    term10 = np.add(term9, np.add(np.divide(X214, np.where(X27 != 0, X27, 1)), X239))

    # 対策: 分母が0の場合、0にする
    result = pd.Series(np.add(term10, np.add(np.maximum(X245, X192), np.maximum(X157, X240))))
    
    # NaNを0に変換する
    return result.fillna(0)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/698/console
# 2179_X08
# add(add(add(add(sin(sin(add(mul(mul(X179, add(sin(max(X245, X192)), div(max(X157, X240), X27))), max(X245, X192)), X245))), mul(X179, add(div(X214, X27), X239))), max(X245, X192)), add(X214, X239)), add(max(X245, X192), add(add(sin(add(mul(mul(X179, add(add(sin(max(X157, X240)), mul(X179, add(add(mul(X179, add(div(X214, X27), X239)), X239), X239))), X239)), max(X245, X192)), X245)), mul(X179, add(add(div(X214, X27), X239), X239))), max(X245, X192))))
############################################################
def create_gp_feature_2179_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X179 = _df['starndard_lead_time']
    X214 = _df['hotelfeature_seasons_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X157 = _df['binning_labels_adr_208_234']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.multiply(X179, X214)
    term2 = np.add(X214, X239)
    term3 = np.divide(np.maximum(X157, X240), X27)
    term4 = np.add(term2, term3)
    term5 = np.multiply(X179, term4)

    # 対策: 分母が0の場合、0にする
    term6 = np.sin(np.add(term5, X245))
    term7 = np.add(X214, X239)
    term8 = np.add(term6, term7)
    term9 = np.add(X245, X192)
    term10 = np.add(term8, term9)

    # 対策: 分母が0の場合、0にする
    term11 = np.divide(X214, np.where(X27 != 0, X27, 1))
    term12 = np.add(term11, X239)
    term13 = np.multiply(X179, term12)
    term14 = np.add(term10, term13)
    term15 = np.add(X245, X192)
    term16 = np.add(term14, term15)

    # 対策: NaNを0に変換する
    result = pd.Series(term16).fillna(0)
    return result


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/708/console
# 4021_X01
# cos(add(X264, add(sub(X211, min(add(X59, X189), X149)), add(min(cos(X245), add(sub(X211, cos(add(X264, add(sub(X211, mul(X27, X245)), add(min(add(sub(X211, sqrt(X26)), X234), min(min(add(X216, X234), sqrt(tan(X127))), X179)), min(add(inv(max(X59, max(X59, X245))), X234), min(sqrt(neg(min(X129, min(tan(X127), min(add(inv(max(X59, X211)), X234), X179))))), abs(X179)))))))), X234)), min(min(add(X216, X234), sqrt(abs(X240))), X179)))))
############################################################
def create_gp_feature_4021_X01(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X211 = _df['hoteldeposit_type_cat_combination_2_target_encoder']
    X59 =  _df['arrival_date_monthdeposit_type_cat_combination_2']
    X189 = _df['domain_is_deposit']
    X149 = _df['binning_labels_adr_0_26']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X157 = _df['binning_labels_adr_208_234']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X214 = _df['hotelfeature_seasons_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X127 = _df['mean_distance_adr']

    term1 = np.add(X59, X189)
    term2 = np.minimum(term1, X149)
    term3 = np.subtract(X211, term2)
    term4 = np.add(X264, term3)
    term5 = np.sin(term4)
    term6 = np.sin(term5)

    # 対策: 分母が0の場合、0にする
    term7 = np.add(X214, X239)
    term8 = np.divide(term7, np.where(X27 != 0, X27, 1))
    term9 = np.add(term8, X240)
    term10 = np.multiply(X179, term9)
    term11 = np.add(term6, term10)

    # 対策: NaNを0に変換する
    term12 = np.maximum(X245, X192)
    term13 = np.add(term11, term12)
    term14 = np.add(X214, X239)
    term15 = np.add(term13, term14)

    # 対策: NaNを0に変換する
    term16 = np.maximum(X245, X192)
    term17 = np.maximum(X157, X240)
    term18 = np.add(term16, term17)
    term19 = np.add(term15, term18)

    # 対策: NaNを0に変換する
    result = pd.Series(np.cos(term19)).fillna(0)
    return result
    
    
############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/741/console
# 2574_X09
# add(add(max(X265, X192), min(X32, max(max(min(X195, min(X195, add(max(X183, X234), max(X183, X234)))), max(X183, X234)), X192))), min(X239, min(min(X239, min(X255, X32)), max(add(add(max(X265, X192), min(X32, max(max(min(X195, add(X234, add(X234, X192))), max(X183, X234)), X192))), min(X239, min(X255, X265))), X192))))
############################################################
def create_gp_feature_2574_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X32 =  _df['lead_time*adr']
    X195 = _df['dev_cancel_singularity_classification']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X183 = _df['relation_market_segment_and_deposit_type_and_customer_type']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']

    term1 = np.maximum(X265, X192)
    term2 = np.minimum(X195, np.add(term1, term1))
    term3 = np.maximum(term1, term1)
    term4 = np.maximum(np.minimum(term2, term3), term1)
    term5 = np.minimum(X32, term4)
    term6 = np.add(term1, term5)
    term7 = np.minimum(X239, np.minimum(X255, X32))
    term8 = np.minimum(term7, term1)
    term9 = np.add(term6, term8)
    term10 = np.minimum(X239, term9)
    term11 = np.add(term6, term10)

    result = pd.Series(term11)
    return result


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/772/console
# 3007_X02
# add(mul(X234, X179), add(sqrt(add(X192, mul(add(X213, add(X192, mul(add(X265, add(add(X192, mul(X234, X97)), mul(X234, X179))), X179))), X240))), add(sin(X212), add(sin(add(X265, add(add(X192, mul(X234, X179)), mul(X234, X179)))), add(sqrt(X255), X265)))))
############################################################
def create_gp_feature_3007_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X213 = _df['hotelyyyy-mm-week_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X212 = _df['hotelcustomer_type_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']

    term1 = np.multiply(X234, X179)
    term2 = np.add(X213, np.add(X192, np.multiply(np.add(X265, np.add(np.add(X192, term1), term1)), X179)))
    term3 = np.sqrt(np.add(X192, term2))
    term4 = np.sin(X212)
    term5 = np.add(X265, np.add(np.add(X192, term1), term1))
    term6 = np.sin(term5)
    term7 = np.sqrt(X255)
    term8 = np.add(term4, np.add(term6, np.add(term7, X265)))
    term9 = np.add(term3, term8)
    term10 = np.add(term1, term9)

    result = pd.Series(term10)
    return result


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/774/console
# 3923_X10
# add(add(add(sin(sin(sin(add(sin(add(sin(add(cos(sin(add(X265, add(cos(X26), X192)))), X179)), X179)), add(add(add(add(X252, add(add(X234, add(add(add(cos(X26), sqrt(X234)), X252), X240)), X265)), add(X179, X222)), X192), cos(X26)))))), X192), add(cos(X26), sqrt(X234))), X265)
############################################################
def create_gp_feature_3923_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X179 = _df['starndard_lead_time']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X222 = _df['arrival_date_monthcustomer_type_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X38  = _df['lead_time*adr*arrival_date_week_number']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X27  = _df['total_of_special_requests']

    # Term 1
    term1_1 = np.cos(X26)
    term1_2 = np.add(X192, term1_1)
    term1_3 = np.sin(term1_2)
    term1_4 = np.add(term1_3, X179)
    term1_5 = np.sin(term1_4)
    term1_6 = np.add(term1_5, X179)

    # Term 2
    term2_1 = np.sin(term1_6)
    term2_2 = np.cos(X26)
    term2_3 = np.add(term2_2, np.sqrt(X234))
    term2_4 = np.add(term2_3, X252)
    term2_5 = np.add(term2_4, X240)
    term2_6 = np.add(term2_5, X265)
    term2_7 = np.add(term2_6, X179)
    term2_8 = np.add(term2_7, X222)
    term2_9 = np.cos(X26)

    # Term 3
    term3_1 = np.sin(term2_8)
    term3_2 = np.add(term3_1, term2_9)

    # Term 4
    term4_1 = np.sin(term3_2)
    term4_2 = np.sin(term4_1)
    term4_3 = np.sin(term4_2)
    term4_4 = np.sin(term4_3)
    term4_5 = np.sin(term4_4)
    term4_6 = np.sin(term4_5)

    # Term 5
    term5_1 = np.sin(X192)
    term5_2 = np.add(term5_1, np.log(np.negative(X255)))

    # Term 6
    term6_1 = np.add(X265, np.negative(X240))

    # Term 7
    term7_1 = np.divide(np.negative(X240), np.add(X192, np.log(np.negative(X255))))

    # Term 8
    term8_1 = np.multiply(term6_1, np.log(X38))
    term8_2 = np.add(term1_1, np.log(np.divide(term8_1, np.add(term1_1, np.log(np.divide(term8_1, np.add(X265, np.divide(term7_1, np.divide(1, np.where(X245!=0, X245, 1))))))))))
    term8_3 = np.subtract(np.sin(term8_2), np.divide(np.negative(X240), np.log(np.negative(X255))))
    term8_4 = np.add(np.log(X38), np.negative(X240))
    term8_5 = np.sin(np.divide(np.log(np.negative(X255)), np.negative(X240)))
    term8_6 = np.sin(np.sin(X234))

    # Term 9
    term9_1 = np.subtract(X27, X245)

    # Result
    result = np.subtract(term8_3, term9_1)
    
    return pd.Series(result).fillna(0)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/780/console
# 3664_X02
# sub(add(X245, add(min(X179, X240), X192)), cos(add(X179, add(add(add(X193, X192), min(X245, sub(min(X179, X240), cos(add(add(add(X245, min(X179, min(X234, X234))), X192), add(X245, min(X179, X234))))))), X234))))
############################################################
def create_gp_feature_3664_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X193 = _df['p_low']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']

    # Term 1
    term1_1 = np.minimum(X179, X240)
    term1_2 = np.add(X245, np.add(term1_1, X192))

    # Term 2
    term2_1 = np.add(X193, X192)
    term2_2 = np.add(X245, np.add(X179, np.add(np.add(np.add(X245, np.minimum(X179, np.minimum(X234, X234))), X192), np.add(X245, np.minimum(X179, X234)))))
    term2_3 = np.cos(np.add(X179, term2_2))
    term2 = np.subtract(term2_1, term2_3)

    # Result
    result = np.subtract(term1_2, term2)
    return pd.Series(result).fillna(0)
    
############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/852/console
# 4663_X08
# sub(mul(neg(max(sqrt(abs(X234)), X238)), sqrt(sub(inv(log(X128)), sqrt(X265)))), max(X238, max(max(max(abs(abs(X17)), mul(neg(max(abs(X17), X238)), sqrt(mul(sqrt(X195), sin(max(sin(sub(abs(X234), mul(sqrt(X195), neg(X240)))), X238)))))), sin(max(sin(sub(abs(X234), mul(sqrt(X265), neg(X240)))), X238))), max(neg(X240), X238))))
############################################################
def create_gp_feature_4663_X08(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X238 = _df['countrydeposit_type_cat_combination_2_target_encoder']
    X128 = _df['binning_labels_lead_time']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X195 = _df['dev_cancel_singularity_classification']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']

    # Convert categorical column to numeric
    X128_numeric = X128.cat.codes

    term1 = np.sqrt(np.abs(X234))
    term2 = np.maximum(term1, X238)
    term3 = np.negative(term2)
    
    # Handle division by zero
    term4 = np.log(X128_numeric.replace(0, np.nan))
    term5 = np.where(term4 != 0, 1 / term4, 0)
    
    term6 = np.sqrt(X265)
    term7 = np.sqrt(np.abs(term5))
    term8 = np.subtract(term5, term7)
    
    # Handle square root of negative number
    term9 = np.sqrt(np.abs(term8))
    
    term10 = np.multiply(term3, term9)
    term11 = np.abs(X17)
    term12 = np.abs(term11)
    term13 = np.maximum(term12, term11)
    term14 = np.maximum(term13, X238)
    term15 = np.negative(term14)
    term16 = np.sqrt(X195)
    term17 = np.multiply(term15, term16)
    term18 = np.sqrt(np.abs(term17))
    term19 = np.abs(X234)
    term20 = np.sqrt(X195)
    term21 = np.negative(X240)
    term22 = np.multiply(term20, term21)
    term23 = np.subtract(term19, term22)
    term24 = np.sin(term23)
    term25 = np.maximum(term24, X238)
    term26 = np.sin(term25)
    term27 = np.maximum(term18, term26)
    term28 = np.abs(term27)
    term29 = np.maximum(term28, term27)
    term30 = np.sin(term29)
    term31 = np.sqrt(X265)
    term32 = np.negative(X240)
    term33 = np.multiply(term31, term32)
    term34 = np.subtract(term19, term33)
    term35 = np.sin(term34)
    term36 = np.maximum(term35, X238)
    term37 = np.sin(term36)
    term38 = np.maximum(term30, term37)
    term39 = np.negative(X240)
    term40 = np.maximum(term39, X238)
    term41 = np.maximum(term38, term40)
    term42 = np.subtract(term10, term41)

    # Handle NaN values in the final result
    term42 = term42.replace([np.inf, -np.inf, np.nan], 0)

    return pd.Series(term42)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/858/console
# 791_X07
# add(sin(abs(X202)), add(add(sqrt(X179), abs(sin(add(sqrt(X179), add(sin(add(X234, X211)), add(sin(sin(add(X234, X211))), add(sqrt(X179), add(add(X240, add(X157, X213)), X245)))))))), add(sin(sin(sin(sin(add(X234, abs(X202)))))), add(add(X240, add(X202, sin(sin(add(X234, X211))))), X245))))
############################################################
def create_gp_feature_791_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X202 = _df['customer_type_target_encoder']
    X179 = _df['starndard_lead_time']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X211 = _df['hoteldeposit_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X157 = _df['binning_labels_adr_208_234']
    X213 = _df['hotelyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']

    term1 = np.sin(np.abs(X202))
    term2 = np.add(np.sqrt(X179), np.abs(np.sin(np.add(np.sqrt(X179), np.add(np.sin(np.add(X234, X211)), np.add(np.sin(np.sin(np.add(X234, X211))), np.add(np.sqrt(X179), np.add(np.add(X240, np.add(X157, X213)), X245))))))))
    term3 = np.sin(np.abs(X202))
    term4 = np.sin(np.sin(np.sin(np.sin(np.add(X234, np.abs(X202))))))
    term5 = np.add(np.add(X240, np.add(X202, np.sin(np.sin(np.add(X234, X211))))), X245)
    term6 = np.add(term1, np.add(term2, np.add(term3, term4)))
    result = np.add(term6, term5)

    return pd.Series(result)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/875/console
# 2699_X04
# cos(sub(mul(mul(X179, sin(add(sin(add(X192, add(sin(add(sin(add(add(X238, X234), X213)), X234)), sub(X184, X26)))), X234))), sin(add(add(X234, X234), X234))), neg(add(X265, add(X192, sin(add(X192, add(X238, X234))))))))
############################################################
def create_gp_feature_2699_X04(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X238 = _df['countrydeposit_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X213 = _df['hotelyyyy-mm-week_cat_combination_2_target_encoder']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X26 =  _df['required_car_parking_spaces']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']

    term1 = np.add(X238, X234)
    term2 = np.add(term1, X213)
    term3 = np.sin(term2)
    term4 = np.add(term3, X234)
    term5 = np.sin(term4)
    term6 = np.subtract(X184, X26)
    term7 = np.add(term5, term6)
    term8 = np.sin(term7)
    term9 = np.add(term8, X234)
    term10 = np.sin(term9)
    term11 = np.multiply(X179, term10)
    term12 = np.multiply(term11, X234)
    term13 = np.add(X234, X234)
    term14 = np.add(term13, X234)
    term15 = np.sin(term14)
    term16 = np.add(X192, X238)
    term17 = np.add(term16, X234)
    term18 = np.sin(term17)
    term19 = np.add(X192, term18)
    term20 = np.add(X265, term19)
    term21 = np.negative(term20)
    term22 = np.subtract(term12, term15)
    term23 = np.cos(term22)

    return pd.Series(term23)

    
############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/879/console
# 3398_X10
# add(min(min(X227, X234), mul(add(min(X240, X179), X265), X179)), max(inv(X183), max(X192, min(X240, min(X35, max(X149, mul(add(add(X234, min(X227, min(X240, max(X192, min(X240, max(X149, mul(add(X227, min(X227, abs(X193))), mul(add(min(min(X227, X234), X179), min(X227, X234)), X179)))))))), min(X227, X234)), min(min(X227, X234), X234))))))))
############################################################
def create_gp_feature_3398_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X227 = _df['mealdistribution_channel_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X183 = _df['relation_market_segment_and_deposit_type_and_customer_type']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X35 =  _df['arrival_date_week_number']
    X149 = _df['binning_labels_adr_0_26']
    X193 = _df['p_low']

    term1 = np.minimum(X227, X234)
    term2 = np.minimum(X240, X179)
    term3 = np.add(term2, X265)
    term4 = np.multiply(term3, X179)
    term5 = np.minimum(term1, term4)
    term6 = np.where(X183 != 0, 1 / X183, 0)
    term7 = np.maximum(term6, X192)
    term8 = np.minimum(X240, X35)
    term9 = np.maximum(X149, term8)
    term10 = np.abs(X193)
    term11 = np.minimum(X227, term10)
    term12 = np.add(X227, term11)
    term13 = np.minimum(term1, X234)
    term14 = np.add(term13, term12)
    term15 = np.multiply(term14, X179)
    term16 = np.maximum(term9, term15)
    term17 = np.minimum(X240, term16)
    term18 = np.minimum(term1, term17)
    term19 = np.maximum(term7, term18)
    term20 = np.add(term5, term19)

    return pd.Series(np.cos(term20))


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/895/console
# 372_X10
# add(min(add(X234, X234), min(abs(X255), min(X240, sqrt(min(X253, X240))))), add(min(X125, min(min(min(mul(add(X193, X234), add(X234, min(X234, X179))), X179), min(X179, X179)), X39)), add(X265, min(min(min(min(X240, add(X234, X234)), X179), min(abs(X255), min(X240, add(X193, X234)))), X39))))
############################################################
def create_gp_feature_372_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X253 = _df['distribution_channelyyyy-mm-week_cat_combination_2_target_encoder']
    X125 = _df['mean_distance_stays_in_week_nights']
    X179 = _df['starndard_lead_time']
    X39 =  _df['lead_time*adr*arrival_date_day_of_month']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X193 = _df['p_low']

    term1 = np.add(X234, X234)
    term2 = np.abs(X255)
    term3 = np.minimum(X253, X240)
    term4 = np.sqrt(term3)
    term5 = np.minimum(X240, term4)
    term6 = np.minimum(term2, term5)
    term7 = np.minimum(term1, term6)
    term8 = np.minimum(X125, X179)
    term9 = np.minimum(X179, X179)
    term10 = np.minimum(term8, term9)
    term11 = np.minimum(term10, X39)
    term12 = np.add(X193, X234)
    term13 = np.add(X234, X179)
    term14 = np.minimum(X234, term13)
    term15 = np.add(X234, term14)
    term16 = np.multiply(term12, term15)
    term17 = np.minimum(term16, X179)
    term18 = np.minimum(term11, term17)
    term19 = np.add(X234, X234)
    term20 = np.minimum(X240, term19)
    term21 = np.abs(X255)
    term22 = np.add(X193, X234)
    term23 = np.minimum(X240, term22)
    term24 = np.minimum(term21, term23)
    term25 = np.minimum(X179, term24)
    term26 = np.minimum(term20, term25)
    term27 = np.minimum(term26, X39)
    term28 = np.add(X265, term27)
    term29 = np.add(term18, term28)
    term30 = np.add(term7, term29)

    return pd.Series(term30)   


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/898/console
# 1903_X10
# add(max(X184, max(X192, min(add(X252, sub(add(min(add(X192, tan(X179)), add(min(X234, X240), X265)), X265), cos(X192))), X180))), max(min(min(X234, X245), X240), min(add(min(add(sub(tan(X179), add(max(X192, X234), X265)), sub(add(tan(X179), X265), cos(add(max(X192, X234), X123)))), X234), sub(max(X192, X234), mul(X19, X27))), X180)))
############################################################
def create_gp_feature_1903_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X180 = _df['starndard_adr']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X125 = _df['mean_distance_stays_in_week_nights']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X193 = _df['p_low']
    X19 =  _df['reserved_room_type']
    X27 =  _df['total_of_special_requests']
    X123 = _df['mean_distance_lead_time']

    term1 = np.add(X252, X265)
    term2 = np.tan(X179)
    term3 = np.add(X192, term2)
    term4 = np.add(X234, X179)
    term5 = np.minimum(X234, term4)
    term6 = np.add(X234, term5)
    term7 = np.subtract(term3, term6)
    term8 = np.add(X192, term7)
    term9 = np.minimum(X234, X240)
    term10 = np.add(term9, X265)
    term11 = np.minimum(X234, term10)
    term12 = np.add(term8, term11)
    term13 = np.subtract(term1, term12)
    term14 = np.cos(X192)
    term15 = np.add(term13, term14)
    term16 = np.minimum(X234, X245)
    term17 = np.minimum(X234, X240)
    term18 = np.minimum(term16, term17)
    term19 = np.multiply(X19, X27)
    term20 = np.subtract(X192, term19)
    term21 = np.maximum(X234, term20)
    term22 = np.minimum(X234, term21)
    term23 = np.add(term18, term22)
    term24 = np.minimum(X125, term23)
    term25 = np.minimum(X180, term24)
    term26 = np.maximum(X184, X192)
    term27 = np.maximum(term26, term15)
    term28 = np.add(term27, term25)

    return pd.Series(term28)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/899/console
# 3562_X05
# add(add(max(X17, X265), max(X264, min(min(min(div(inv(X17), X137), min(min(min(div(min(X26, X61), X137), add(min(X195, add(X255, X255)), X234)), div(min(X234, X22), min(X26, X61))), div(min(X234, X22), min(X195, X255)))), div(min(min(X17, min(add(X255, X234), X255)), X265), min(X26, X61))), div(min(X234, X22), min(X195, X255))))), min(min(div(inv(X17), min(X26, X61)), add(X255, X255)), X255))
############################################################
def create_gp_feature_2699_X04(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X17 =  _df['previous_cancellations']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X137 = _df['binning_labels_lead_time_140_160']
    X26 =  _df['required_car_parking_spaces']
    X61 =  _df['arrival_date_monthyyyy-mm-week_cat_combination_2']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X180 = _df['starndard_adr']
    X125 = _df['mean_distance_stays_in_week_nights']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X193 = _df['p_low']
    X179 = _df['starndard_lead_time']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X22  = _df['agent']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X19  = _df['reserved_room_type']
    X27  = _df['total_of_special_requests']

    term1 = np.maximum(X17, X265)
    term2 = np.where(X17 != 0, 1 / X17, 0)
    term3 = np.divide(term2, X137)
    term4 = np.minimum(X26, X61)
    term5 = np.divide(term3, term4)
    term6 = np.add(X255, X255)
    term7 = np.minimum(X234, term6)
    term8 = np.add(X193, term7)
    term9 = np.minimum(X234, X179)
    term10 = np.add(term8, term9)
    term11 = np.multiply(term10, X179)
    term12 = np.minimum(term5, term11)
    term13 = np.minimum(X179, term12)
    term14 = np.minimum(X26, X61)
    term15 = np.divide(term13, term14)
    term16 = np.minimum(X234, X22)
    term17 = np.divide(term15, term16)
    term18 = np.minimum(X234, X255)
    term19 = np.divide(term17, term18)
    term20 = np.minimum(X234, X255)
    term21 = np.add(X252, term20)
    term22 = np.minimum(X264, term21)
    term23 = np.add(term1, term22)
    term24 = np.minimum(X234, X245)
    term25 = np.minimum(X234, X240)
    term26 = np.minimum(term24, term25)
    term27 = np.tan(X179)
    term28 = np.add(X192, term27)
    term29 = np.add(X234, X265)
    term30 = np.maximum(X192, term29)
    term31 = np.add(term28, term30)
    term32 = np.tan(X179)
    term33 = np.add(term31, term32)
    term34 = np.add(X234, term33)
    term35 = np.subtract(term34, term32)
    term36 = np.minimum(X234, term35)
    term37 = np.subtract(X192, term36)
    term38 = np.multiply(X19, X27)
    term39 = np.subtract(term37, term38)
    term40 = np.minimum(term26, term39)
    term41 = np.minimum(X125, term40)
    term42 = np.add(term23, term41)

    return pd.Series(term42)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/909/console
# 231_X04
# add(sin(add(X239, max(X157, X265))), max(div(add(X157, sin(X265)), X110), add(neg(X26), add(max(X192, add(neg(div(log(X110), abs(X128))), sin(add(X234, add(neg(neg(div(log(X110), abs(X128)))), sin(add(X234, X234))))))), max(X192, X265)))))
############################################################
def create_gp_feature_231_X04(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X157 = _df['binning_labels_adr_208_234']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X110 = _df['math_lead_time_amax_groupby_yyyy-mm-week']
    X26 =  _df['required_car_parking_spaces']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X128 = _df['binning_labels_lead_time'].astype('float64')

    term1 = np.add(X239, X157)
    term2 = np.sin(term1)
    term3 = np.add(X157, X265)
    term4 = np.sin(term3)
    term5 = np.add(term3, term4)
    term6 = np.where(X110 != 0, 1 / X110, 0)
    term7 = np.divide(term5, term6)
    term8 = np.negative(X26)
    term9 = np.log(X110)
    term10 = np.abs(X128)
    term11 = np.where(term9 != 0, 1 / term9, 0)
    term12 = np.negative(term11)
    term13 = np.add(X234, term12)
    term14 = np.sin(term13)
    term15 = np.negative(term11)
    term16 = np.negative(term15)
    term17 = np.add(X234, term16)
    term18 = np.sin(term17)
    term19 = np.add(X234, term18)
    term20 = np.sin(term19)
    term21 = np.add(term14, term20)
    term22 = np.maximum(X192, term21)
    term23 = np.add(term8, term22)
    term24 = np.maximum(X192, X265)
    term25 = np.add(term23, term24)
    term26 = np.maximum(term7, term25)
    term27 = np.add(term2, term26)

    return pd.Series(term27)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/911/console
# 1498_X07
# add(add(add(mul(X240, add(add(X245, X234), X234)), min(mul(X179, add(X245, X234)), X240)), min(min(min(inv(X193), X240), mul(X240, add(min(inv(X193), mul(X179, add(add(X245, X234), X234))), X234))), inv(X193))), add(min(div(X265, X27), mul(X179, add(add(X245, X234), X234))), add(X202, add(X202, sin(X192)))))
############################################################
def create_gp_feature_1498_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()

    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X193 = _df['p_low']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X202 = _df['customer_type_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']

    term0 = np.add(X245, X234)
    term1 = np.add(term0, X234)
    term2 = np.multiply(X240, term1)
    term3 = np.add(term2, X234)
    term4 = np.minimum(np.divide(X265, X27), term3)
    term5 = np.add(term4, X202)
    term6 = np.add(term5, np.sin(X192))
    term7 = np.minimum(np.divide(X193, X240), term6)
    term8 = np.minimum(term7, term3)
    term9 = np.multiply(X240, term8)
    term10 = np.minimum(term9, X193)
    term11 = np.add(term10, term6)
    term12 = np.add(term11, X202)
    term13 = np.minimum(term12, X193)
    term14 = np.add(term13, term6)

    return pd.Series(term14)



############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/915/console
# 915_X09
# add(X264, add(X17, add(sqrt(min(add(X219, X249), min(X245, min(min(min(min(sin(min(min(X240, X38), X38)), min(min(sqrt(min(X240, X38)), sqrt(X265)), div(min(X240, X189), min(X26, X265)))), sqrt(X265)), X234), X234)))), add(min(min(sqrt(min(X240, X38)), sqrt(X265)), X234), sin(sin(sub(add(X219, X249), add(add(X234, X245), add(div(min(X240, X189), min(X26, X265)), add(add(X265, sin(add(X265, sin(X195)))), add(add(X264, X255), add(sqrt(sqrt(min(X219, min(X245, min(sqrt(div(X148, X27)), X234))))), min(X265, X17)))))))))))))
############################################################
def create_gp_feature_915_X09(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X219 = _df['arrival_date_monthreserved_room_type_cat_combination_2_target_encoder']
    X249 = _df['distribution_channelreserved_room_type_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X38 =  _df['lead_time*adr*arrival_date_week_number']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X189 = _df['domain_is_deposit']
    X26 =  _df['required_car_parking_spaces']
    X195 = _df['dev_cancel_singularity_classification']
    X148 = _df['binning_labels_lead_time_360_380']
    X27 =  _df['total_of_special_requests']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']

    term1 = np.minimum(X240, X38)
    term2 = np.minimum(term1, X38)
    term3 = np.sin(term2)
    term4 = np.minimum(term3, X234)
    term5 = np.sqrt(term1)
    term6 = np.sqrt(X265)
    term7 = np.minimum(term5, term6)
    term8 = np.minimum(X240, X189)
    term9 = np.minimum(X26, X265)
    term10 = np.divide(term8, term9, out=np.zeros_like(term8), where=term9!=0) 
    term11 = np.minimum(term4, term7)
    term12 = np.sin(term11)
    term13 = X219 + X249
    term14 = term4 + X245
    term15 = term10 + term12
    term16 = X265 + np.sin(X265 + X195)
    term17 = term15 + term16
    term18 = X264 + X255
    term19 = np.sqrt(np.sqrt(np.minimum(X219, np.minimum(X245, np.minimum(np.sqrt(term10), X234)))))
    term20 = np.minimum(X265, X17) 
    term21 = term18 + term19 + term20
    term22 = term17 + term21
    term23 = term13 - term14 - term22
    
    return pd.Series(term23)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/920/console
# 2146_X07
# add(add(div(X256, X26), add(sqrt(max(X256, add(min(X9, X192), X267))), min(min(sqrt(min(sin(add(add(add(sqrt(min(X240, X234)), X234), X234), X234)), sqrt(min(X195, sqrt(sqrt(min(X240, X234))))))), div(X256, X26)), add(X234, X234)))), max(max(add(X234, sin(sqrt(min(min(sin(X255), tan(X192)), X193)))), tan(X192)), tan(X183)))
############################################################
def create_gp_feature_2146_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X256 = _df['reserved_room_typedeposit_type_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X9 =   _df['adults']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X267 = _df['customer_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X193 = _df['p_low']
    X183 = _df['relation_market_segment_and_deposit_type_and_customer_type']

    term1 = np.divide(X256, np.where(X26 != 0, X26, 1e-10))
    term2 = np.minimum(X9, X192)
    term3 = np.add(term2, X267)
    term4 = np.maximum(X256, term3)
    term5 = np.sqrt(term4)
    term6 = np.add(term5, term1)
    term7 = np.sin(np.add(term6, X234))
    term8 = np.minimum(term7, X234)
    term9 = np.sqrt(term8)
    term10 = np.minimum(X240, X234)
    term11 = np.sqrt(term10)
    term12 = np.minimum(term11, term9)
    term13 = np.sqrt(term12)
    term14 = np.minimum(X195, term13)
    term15 = np.sqrt(term14)
    term16 = np.minimum(term15, term1)
    term17 = np.add(X234, X234)
    term18 = np.minimum(term16, term17)
    term19 = np.sin(X255)
    term20 = np.tan(X192)
    term21 = np.minimum(term19, term20)
    term22 = np.minimum(term21, X193)
    term23 = np.sqrt(term22)
    term24 = np.sin(term23)
    term25 = np.add(X234, term24)
    term26 = np.maximum(term25, term20)
    term27 = np.tan(X183)
    term28 = np.maximum(term26, term27)
    term29 = np.where(np.isnan(term28), 0, term28)

    return pd.Series(term29)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/937/console
# 3527_X07
# add(max(X192, min(min(max(X238, sqrt(X234)), div(X265, X238)), add(X265, X240))), max(X192, max(X192, min(min(X25, max(min(max(max(X192, min(min(max(X238, sqrt(X234)), div(X265, X238)), add(X265, X240))), min(cos(X27), min(X25, min(X25, min(add(X265, sqrt(X234)), max(min(max(X238, min(cos(X27), add(X265, log(X22)))), min(add(X265, sqrt(X234)), add(X265, X179))), X182)))))), max(X179, X192)), X182)), add(X265, log(X22))))))
############################################################
def create_gp_feature_3527_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X238 = _df['countrydeposit_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X25 =  _df['adr']
    X27 =  _df['total_of_special_requests']
    X22 =  _df['agent']
    X179 = _df['dev_value_counts_distribution_channel']
    X182 = _df['relation_market_segment_and_deposite_type']

    term1 = np.sqrt(X234)
    term2 = np.maximum(X238, term1)
    term3 = np.divide(X265, np.where(X238 != 0, X238, 1e-10))
    term4 = np.maximum(term2, term3)
    term5 = np.add(X265, X240)
    term6 = np.minimum(term4, term5)
    term7 = np.maximum(X192, term6)
    term8 = np.cos(X27)
    term9 = np.minimum(term8, X25)
    term10 = np.add(X265, np.log(np.where(X22 != 0, X22, 1e-10)))
    term11 = np.sqrt(X234)
    term12 = np.add(X265, term11)
    term13 = np.minimum(term10, term12)
    term14 = np.minimum(term9, term13)
    term15 = np.maximum(term7, term14)
    term16 = np.maximum(X179, X192)
    term17 = np.maximum(term15, term16)
    term18 = np.maximum(X192, term17)
    term19 = np.add(X265, np.log(np.where(X22 != 0, X22, 1e-10)))
    term20 = np.maximum(X192, term19)

    return pd.Series(term20)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/954/console
# 3684_X07
# add(min(min(max(add(X234, add(add(max(neg(X46), add(add(X234, X146), X240)), add(max(X234, X17), add(X234, add(X234, add(X234, log(X265)))))), X245)), X17), add(X179, X240)), add(X179, X179)), max(X265, X17))
############################################################
def create_gp_feature_3684_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X46 =  _df['hoteldistribution_channel_cat_combination_2']
    X146 = _df['binning_labels_lead_time_320_340']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']

    term1 = np.negative(X46)
    term2 = np.add(X234, X146)
    term3 = np.add(term2, X240)
    term4 = np.maximum(term1, term3)
    term5 = np.add(X234, X17)
    term6 = np.add(X234, X265)
    term7 = np.log(np.where(X265 != 0, X265, 1e-10))
    term8 = np.add(X234, term7)
    term9 = np.add(X234, term8)
    term10 = np.maximum(term5, term9)
    term11 = np.add(term4, term10)
    term12 = np.maximum(term11, X245)
    term13 = np.minimum(term12, X17)
    term14 = np.add(X179, X240)
    term15 = np.minimum(term13, term14)
    term16 = np.add(X179, X179)
    term17 = np.minimum(term15, term16)
    term18 = np.maximum(X265, X17)
    term19 = np.add(term17, term18)

    return pd.Series(term19)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/964/console
# 3796_X07
# add(add(min(X265, X255), max(max(max(sub(X141, X217), max(sin(X234), X17)), X17), X183)), max(min(X234, X208), max(max(add(min(X265, X255), max(max(sin(X234), X17), X183)), sin(add(min(X265, X255), max(max(X234, X183), X183)))), X265)))
############################################################
def create_gp_feature_3796_X07(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X141 = _df['binning_labels_lead_time_220_240']
    X217 = _df['arrival_date_monthmarket_segment_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X183 = _df['relation_market_segment_and_deposit_type_and_customer_type']
    X208 = _df['hoteldistribution_channel_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']

    term1 = np.minimum(X265, X255)
    term2 = np.subtract(X141, X217)
    term3 = np.sin(X234)
    term4 = np.maximum(term3, X17)
    term5 = np.maximum(term2, term4)
    term6 = np.maximum(term5, X17)
    term7 = np.maximum(term6, X183)
    term8 = np.add(term1, term7)
    term9 = np.minimum(X234, X208)
    term10 = np.sin(term8)
    term11 = np.maximum(term10, X265)
    term12 = np.maximum(term9, term11)
    term13 = np.add(term8, term12)

    return pd.Series(term13)



############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/966/console
# 478_X10
# add(min(min(min(X239, max(max(X192, min(div(X178, X129), sqrt(min(X35, min(min(X255, sqrt(sqrt(X193))), min(X234, min(X240, min(X255, min(X234, div(X178, X193)))))))))), X195)), min(X35, min(X234, X255))), X202), max(X184, max(X192, min(div(X178, X129), sqrt(min(X35, min(min(X255, sqrt(sqrt(X193))), min(X234, min(X240, min(X255, sqrt(min(X35, max(sqrt(log(X120)), max(X192, X234))))))))))))))
############################################################
def create_gp_feature_478_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X178 = _df['dev_value_counts_adr']
    X129 = _df['binning_labels_adr']
    X35 =  _df['adr*arrival_date_week_number']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X193 = _df['p_low']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X195 = _df['dev_cancel_singularity_classification']
    X202 = _df['customer_type_target_encoder']
    X184 = _df['relation_distribution_channel_and_deposit_type_and_customer_type']
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X120 = _df['math_adr_sum_groupby_yyyy-mm-week']

    term1 = np.minimum(X239, X192)
    term2 = np.divide(X178, np.where(X129 != 0, X129, 1e-10))
    term3 = np.minimum(term2, X35)
    term4 = np.sqrt(term3)
    term5 = np.minimum(X255, term4)
    term6 = np.sqrt(term5)
    term7 = np.minimum(X255, term6)
    term8 = np.minimum(X234, term7)
    term9 = np.minimum(X240, term8)
    term10 = np.divide(X178, np.where(X193 != 0, X193, 1e-10))
    term11 = np.minimum(X255, term10)
    term12 = np.minimum(X234, term11)
    term13 = np.minimum(term9, term12)
    term14 = np.maximum(term1, term13)
    term15 = np.maximum(term14, X195)
    term16 = np.minimum(X35, X234)
    term17 = np.minimum(term16, X255)
    term18 = np.minimum(term15, term17)
    term19 = np.add(term18, X202)
    term20 = np.maximum(X184, X192)
    term21 = np.log(np.where(X120 != 0, X120, 1e-10))
    term22 = np.sqrt(term21)
    term23 = np.maximum(term22, X192)
    term24 = np.maximum(term23, X234)
    term25 = np.minimum(X35, term24)
    term26 = np.sqrt(term25)
    term27 = np.minimum(X255, term26)
    term28 = np.minimum(X234, term27)
    term29 = np.minimum(X240, term28)
    term30 = np.sqrt(term29)
    term31 = np.minimum(X255, term30)
    term32 = np.minimum(X234, term31)
    term33 = np.minimum(term32, term2)
    term34 = np.sqrt(term33)
    term35 = np.minimum(term34, X35)
    term36 = np.minimum(X255, term35)
    term37 = np.sqrt(term36)
    term38 = np.minimum(X255, term37)
    term39 = np.minimum(X234, term38)
    term40 = np.minimum(X240, term39)
    term41 = np.minimum(term40, term10)
    term42 = np.maximum(term20, term41)
    term43 = np.add(term19, term42)

    return pd.Series(term43)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/968/console
# 4738_X02
# add(abs(X265), add(mul(X258, X17), add(tan(X239), sub(min(X234, X255), sin(mul(sub(mul(sub(neg(X234), sqrt(min(X234, sqrt(sin(min(X234, sqrt(X240))))))), sin(add(sqrt(sqrt(sqrt(min(X234, X240)))), sqrt(sub(mul(neg(X26), X83), X240))))), X245), X179))))))
############################################################
def create_gp_feature_4738_X02(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X265 = df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X258 = df['reserved_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X17 =  df['previous_cancellations']
    X239 = df['countrycustomer_type_cat_combination_2_target_encoder']
    X234 = df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X178 = df['dev_value_counts_adr']
    X129 = df['binning_labels_adr']
    X35 =  df['adr*arrival_date_week_number']
    X193 = df['p_low']
    X240 = df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X26 =  df['required_car_parking_spaces']
    X83 =  df['market_segmentdeposit_type_cat_combination_2']
    X245 = df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X179 = df['starndard_lead_time']
    X120 = df['math_adr_sum_groupby_yyyy-mm-week']

    term1 = np.abs(X265)
    term2 = np.multiply(X258, X17)
    term3 = np.tan(X239)
    term4 = np.negative(X234)
    term5 = np.minimum(X234, X240)
    term6 = np.sqrt(term5)
    term7 = np.sin(term6)
    term8 = np.minimum(X234, term7)
    term9 = np.sqrt(term8)
    term10 = np.minimum(X234, X255)
    term11 = np.minimum(term10, term9)
    term12 = np.subtract(term4, term11)
    term13 = np.multiply(term12, term12)
    term14 = np.sin(term13)
    term15 = np.sqrt(term14)
    term16 = np.sqrt(term15)
    term17 = np.sqrt(term16)
    term18 = np.minimum(X234, X240)
    term19 = np.sqrt(term18)
    term20 = np.subtract(term17, term19)
    term21 = np.negative(X26)
    term22 = np.multiply(term21, X83)
    term23 = np.subtract(term22, X240)
    term24 = np.sqrt(np.abs(term23))  # Applying absolute function to handle NaN in sqrt
    term25 = np.sin(term24)
    term26 = np.subtract(term20, term25)
    term27 = np.multiply(term26, X245)
    term28 = np.sin(term27)
    term29 = np.subtract(term28, X179)
    term30 = np.minimum(X234, X255)
    term31 = np.subtract(term30, term29)
    term32 = np.add(term3, term31)
    term33 = np.add(term2, term32)
    term34 = np.add(term1, term33)

    return pd.Series(term34)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/976/console
# 3506_X10
# add(X265, add(inv(X192), log(add(mul(add(max(cos(X27), X240), min(max(X140, X240), log(X128))), mul(add(cos(X27), mul(log(X128), X252)), min(min(X252, min(min(log(X128), abs(X41)), log(X128))), min(log(X128), min(max(X140, X240), X234))))), X201))))
############################################################
def create_gp_feature_3506_X10(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X265 = _df['deposit_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X258 = _df['reserved_room_typeyyyy-mm-week_cat_combination_2_target_encoder']
    X17 =  _df['previous_cancellations']
    X239 = _df['countrycustomer_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X178 = _df['dev_value_counts_adr']
    X129 = _df['binning_labels_adr']
    X35 =  _df['adr*arrival_date_week_number']
    X193 = _df['p_low']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X26 =  _df['required_car_parking_spaces']
    X83 =  _df['market_segmentdeposit_type_cat_combination_2']
    X245 = _df['market_segmentdeposit_type_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X120 = _df['math_adr_sum_groupby_yyyy-mm-week']
    X27  = _df['total_of_special_requests']
    X140 = _df['binning_labels_lead_time_200_220']
    X128 = _df['binning_labels_lead_time']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X41  = _df['adr*arrival_date_week_number*arrival_date_day_of_month']

    term1 = np.abs(X265)
    term2 = np.divide(1, np.where(X192 != 0, X192, 1e-10))
    term3 = np.cos(X27)
    term4 = np.maximum(term3, X240)
    term5 = np.maximum(X140, X240)
    term6 = np.log(np.where(X128 != 0, X128, 1e-10))
    term7 = np.maximum(term5, term6)
    term8 = np.minimum(term4, term7)
    term9 = np.multiply(term8, X258)
    term10 = np.log(np.where(X128 != 0, X128, 1e-10))
    term11 = np.sqrt(term10)
    term12 = np.sqrt(term11)
    term13 = np.sqrt(term12)
    term14 = np.minimum(X234, X240)
    term15 = np.sqrt(term14)
    term16 = np.minimum(X255, term15)
    term17 = np.minimum(X234, term16)
    term18 = np.minimum(X240, term17)
    term19 = np.minimum(X255, term18)
    term20 = np.divide(X178, np.where(X193 != 0, X193, 1e-10))
    term21 = np.minimum(term19, term20)
    term22 = np.sin(term21)
    term23 = np.subtract(term9, term22)
    term24 = np.multiply(term23, X252)
    term25 = np.log(np.where(X128 != 0, X128, 1e-10))
    term26 = np.abs(X41)
    term27 = np.minimum(term25, term26)
    term28 = np.log(np.where(X128 != 0, X128, 1e-10))
    term29 = np.minimum(term27, term28)
    term30 = np.log(np.where(X128 != 0, X128, 1e-10))
    term31 = np.maximum(X140, X240)
    term32 = np.minimum(term30, term31)
    term33 = np.minimum(term29, term32)
    term34 = np.minimum(term24, term33)
    term35 = np.add(term2, term34)
    term36 = np.add(term1, term35)
    term37 = term36 - np.floor(term36)  # 少数部分だけを取り出す

    return pd.Series(term37)


############################################################
# http://localhost:3800/job/MUFG%20%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B32023/job/CreateGPFeature/1008/console
# 2964_X06
# sqrt(sqrt(max(X256, add(add(sin(sin(div(sin(add(X234, X255)), sin(X179)))), div(X226, X27)), add(log(mul(max(X256, add(sin(div(div(X226, add(X255, X255)), mul(add(sin(X226), add(sin(X179), max(X192, X240))), X264))), add(add(log(sin(add(add(X234, X255), X234))), X192), add(sin(div(div(X226, add(X255, X255)), mul(max(X256, add(sin(add(X234, div(X226, max(X192, X240)))), add(sin(X179), max(X192, X240)))), X264))), add(add(log(sin(add(add(X182, add(X234, X255)), X234))), X192), max(X192, X240)))))), X264)), sin(sin(X179)))))))
############################################################
def create_gp_feature_2964_X06(df: pd.DataFrame) -> pd.Series:
    _df = df.copy()
    
    X256 = _df['reserved_room_typedeposit_type_cat_combination_2_target_encoder']
    X234 = _df['countrymarket_segment_cat_combination_2_target_encoder']
    X255 = _df['reserved_room_typeassigned_room_type_cat_combination_2_target_encoder']
    X179 = _df['starndard_lead_time']
    X226 = _df['mealmarket_segment_cat_combination_2_target_encoder']
    X27 =  _df['total_of_special_requests']
    X140 = _df['binning_labels_lead_time_200_220']
    X240 = _df['countryyyyy-mm-week_cat_combination_2_target_encoder']
    X128 = _df['binning_labels_lead_time']
    X252 = _df['distribution_channelcustomer_type_cat_combination_2_target_encoder']
    X192 = _df['domain_previous_booking_minus_canceled_and_notcanceled']
    X201 = _df['deposit_type_target_encoder']
    X264 = _df['deposit_typecustomer_type_cat_combination_2_target_encoder']

    term1 = np.sin(X234)
    term2 = np.add(term1, X255)
    term3 = np.sin(term2)
    term4 = np.sin(term3)
    term5 = np.divide(term4, np.where(X179 != 0, X179, 1e-10))
    term6 = np.sin(term5)
    term7 = np.add(X234, term6)
    term8 = np.sin(term7)
    term9 = np.divide(X226, np.where(X27 != 0, X27, 1e-10))
    term10 = np.add(term8, term9)
    term11 = np.maximum(X256, term10)
    term12 = np.sin(X226)
    term13 = np.add(term12, X234)
    term14 = np.sin(term13)
    term15 = np.add(term14, X179)
    term16 = np.maximum(X192, X240)
    term17 = np.add(term15, term16)
    term18 = np.multiply(term17, X264)
    term19 = np.log1p(np.abs(X128.astype(float)))  # Using log1p to handle NaN and infinite values
    term20 = np.sin(term19)
    term21 = np.add(term20, X234)
    term22 = np.add(term21, X234)
    term23 = np.sin(term22)
    term24 = np.log1p(np.abs(term23))  # Using log1p to handle NaN and infinite values
    term25 = np.add(term24, X192)
    term26 = np.maximum(X192, X240)
    term27 = np.add(term25, term26)
    term28 = np.log1p(np.abs(term27))  # Using log1p to handle NaN and infinite values
    term29 = np.sin(term28)
    term30 = np.add(term18, term29)
    term31 = np.log1p(np.abs(term30))  # Using log1p to handle NaN and infinite values
    term32 = np.add(term31, X201)
    term33 = np.sqrt(np.abs(term32))  # Using sqrt to handle NaN and infinite values
    term34 = np.sqrt(np.abs(term33))  # Using sqrt to handle NaN and infinite values

    return pd.Series(term34)








    













