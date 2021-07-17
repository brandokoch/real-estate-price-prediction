from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

models={
    "baseline": DummyRegressor(),
    "dt": DecisionTreeRegressor(),
    "rf": RandomForestRegressor(),
    "lin_reg":LinearRegression(),
    "kn":KNeighborsRegressor(),
    "svr":SVR(),
    "xgb":xgb.XGBRegressor(),
    "lgb":lgb.LGBMRegressor(),
    "nn":MLPRegressor(),
    "rf_best":RandomForestRegressor(max_depth= 30, max_features=0.5, min_samples_leaf= 2, min_samples_split= 2, n_estimators=120),
}
