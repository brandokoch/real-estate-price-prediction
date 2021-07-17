from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def get_pipeline_v1():
    num_attribs=['lat', 'lng','year_built', 'bedroom_cnt',
           'full_bathroom_cnt', 'parking_cnt', 'partial_bathroom_cnt', 'm2']

    cat_attribs=['type']

    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('scaler',StandardScaler()),
    ])

    cat_pipeline=Pipeline([
        ('encoder',OneHotEncoder(handle_unknown='ignore')),
    ])

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',cat_pipeline,cat_attribs),
    ])

    return full_pipeline

def get_pipeline_v2():
    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('scaler',StandardScaler()),
    ])

    cat_pipeline=Pipeline([
        ('encoder',OneHotEncoder(handle_unknown='ignore')),
    ])

    num_attribs=['lat', 'lng', 'year_built', 'bedroom_cnt', 'full_bathroom_cnt',
        'partial_bathroom_cnt', 'floorsize_m2', 'floorsize_m2_per_bedroom',
        'lot_size_m2', 'sale_year', 'sale_month', 'sale_day']

    cat_attribs=['type']

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',cat_pipeline,cat_attribs),
    ])

    return full_pipeline

preprocessors={
    'preprocessor_v1':get_pipeline_v1(),
    'preprocessor_v2':get_pipeline_v2(),
}