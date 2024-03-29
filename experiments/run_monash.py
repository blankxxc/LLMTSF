import os
import pickle
import numpy as np
import pandas as pd
from data.monash import get_datasets
from sklearn.model_selection import train_test_split
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
import numpy as np
import openai
CUDA_VISIBLE_DEVICES = 0,1,2
# openai.log = "debug"
openai.api_key = "sk-y4UpRWH6BgZ3CeABx0jw3jrDHjpDMVhLp8G1oPQ9PAKTR9cS"
#openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_base = "https://api.chatanywhere.tech"
# Specify the hyperparameter grid for each model
gpt3_hypers = dict(
    temp=0.7,
    alpha=0.9,
    beta=0,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

model_hypers = {
    'gpt-4': {'model': 'gpt-4', **gpt3_hypers},
    'gpt-3.5-turbo': {'model': 'gpt-3.5-turbo', **gpt3_hypers},
    'gpt-3.5-turbo-instruct': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    'llama-70b': {'model': 'llama-70b', **llama_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'gpt-4': get_llmtime_predictions_data,
    'gpt-3.5-turbo': get_llmtime_predictions_data,
    'gpt-3.5-turbo-instruct': get_llmtime_predictions_data,
    'llama-7b': get_llmtime_predictions_data,
    'llama-70b': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci','gpt-3.5-turbo-instruct', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/AAPL'
os.makedirs(output_dir, exist_ok=True)

models_to_run = [
#      'gpt-4',
#	'gpt-3.5-turbo',
    'gpt-3.5-turbo-instruct',
#    'llama-7b',
    # 'llama-70b',
]
#datasets_to_run =  [
#    "weather", "covid_deaths", "solar_weekly", "tourism_monthly", "australian_electricity_demand", "pedestrian_counts",
#    "traffic_hourly", "hospital", "fred_md", "tourism_yearly", "tourism_quarterly", "us_births",
#    "nn5_weekly", "traffic_weekly", "saugeenday", "cif_2016", "bitcoin", "sunspot", "nn5_daily"
#]
#datasets_to_run =  ["weather"]
datasets_to_run =  ["AAPL"]
max_history_len = 1000
datasets = get_datasets()
for dsname in datasets_to_run:
#    print(f"Starting {dsname}")
#    data = datasets[dsname]
#    train, test = data
#    train = [x[-max_history_len:] for x in train]
#    train=train[:100]
#    test=test[:100]
#    print(len(train[0]))
#    print(len(test[0]))

## 读取数据

    file_path = '/home/xuncan/tsf/data/AAPL_process.csv'
    data = pd.read_csv(file_path)



# 设置滑动窗口大小和预测长度
    window_size = 384
    predict_length = 24

# 仅提取开盘价（Open）作为特征
    open_prices = data['Open'].values

# 初始化特征和标签列表
    features = []
    labels = []

# 使用滑动窗口方法生成特征和标签
    for i in range(len(open_prices) - window_size - predict_length):
        features.append(open_prices[i:i+window_size])
        labels.append(open_prices[i+window_size:i+window_size+predict_length])
    train=features
    test=labels



    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in models_to_run:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_gpt(model) else False
        num_samples = 20
        
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            medians = preds['median']
            targets = np.array(test)
            maes = np.mean(np.abs(medians - targets), axis=1) # (num_series)        
            preds['maes'] = maes
            preds['mae'] = np.mean(maes)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    print(f"Finished {dsname}")
    