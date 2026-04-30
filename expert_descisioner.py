"""
data_regime0 = merged[merged["state"] == 0]
data_regime1 = merged[merged["state"] == 1]
data_regime2 = merged[merged["state"] == 2]
data_regime3 = merged[merged["state"] == 3]
print(f"Shape 0: {data_regime0.shape} | Std : {data_regime0["ret_2"].std()}")
print(f"Shape 1: {data_regime1.shape} | Std : {data_regime1["ret_2"].std()}")
print(f"Shape 2: {data_regime2.shape} | Std : {data_regime2["ret_2"].std()}")
print(f"Shape 3: {data_regime3.shape} | Std : {data_regime3["ret_2"].std()}")

profile_cols = [
    "ret_20", "ret_60",
    "vol_20", "vol_60",
    "drawdown_20", "market_drawdown_20",
    "dist_ma_20", "ma_ratio_20_60",
    "vix_level", "market_breadth",
    "amihud_illiq", "vol_ratio_5_20"
]

summary = merged.groupby("state")[profile_cols].agg(["mean", "std", "median"])
print(summary)
STATE_TO_REGIME = {
    1: 1,  # low vol / range
    2: 2,  # high vol / stress
    3: 3,  # trend fort
    0: 4,  # instable
}

#X_t = torch.tensor(X, dtype=torch.float32)
#y_t = torch.tensor(Y, dtype=torch.float32)



"""