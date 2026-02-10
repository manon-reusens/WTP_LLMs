import joblib
import pandas as pd
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Gather the results from the batches.")
    parser.add_argument('--model', type=str, choices=['llama-3.1-8B','llama-3.2-3B','llama-3.3-70B','qwen2.5-7B','gpt-5-mini','gpt-4o','gemini-3-pro','haiku'] ,help="Name of the model you want to gather the multinomial logit models for")
    parser.add_argument('--which',type=str,choices=["first","second", "both"],help='which of the answers you train the models on; first & second is to make one model and compare results for order bias')
    parser.add_argument('--prompt',type=str,default='all',help='which prompt to calculate the model for')
    parser.add_argument('--currency',type=str,default='HK',help='which currency the experiments were in')
    args = parser.parse_args()
    return args.model,args.which,args.prompt,args.currency


if __name__ == "__main__":
    model_name,which,prompt,curr = parse_args()
    path='results/multinomial_models/'
    model=joblib.load(f"{path}/{model_name}/multinomial_logit_model_{prompt}_{curr}_dilemmas_{which}.joblib")
    scaler=joblib.load(f"{path}/{model_name}/standardscaler/standardscaler_{prompt}_{curr}_dilemmas_{which}.joblib")
    print(model.summary())

    price_name = "price_per_night"
    coeff = model.coeff_
    stdevs=np.sqrt(scaler.var_)
    names = list(model.coeff_names)
    price_coef = coeff[-1]
    stdev_price=stdevs[-1]

    model_wtp = {}
    for i, name in enumerate(names):
        if (name==price_name) | (i==0):
            continue  # skip price itself & the intercept
        stdef_coeff=stdevs[i-1]
        wtp_value = (coeff[i] *stdev_price)/ ((-price_coef)*stdef_coeff)
        model_wtp[name] = wtp_value

    print(model_wtp)
    wtp_df = pd.DataFrame.from_dict(model_wtp, orient='index')
    # rows: models, columns: coefficient names; NaN where not present in a model

    print(wtp_df)