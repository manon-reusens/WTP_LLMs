import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, auc
from xlogit import MultinomialLogit
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Gather the results from the batches.")
    parser.add_argument('--model', type=str, choices=['llama-3.1-8B','llama-3.2-3B','llama-3.3-70B','qwen2.5-7B','gpt-5-mini','gpt-4o','gemini-3-pro','haiku'] ,help="Name of the model you want to gather the multinomial logit models for")
    parser.add_argument('--temperature', type=str, required=True, help="temperature value used in the experiments")
    parser.add_argument('--template',type=str,choices=['hotel_room'],help='the prompt you are evaluating')
    parser.add_argument('--runs',type=str,help='amount of runs')
    parser.add_argument('--which',type=str,choices=["first","second", "both"],help='which of the answers you train the models on; first & second is to make one model and compare results for order bias')
    parser.add_argument('--prompt',type=str,default='all',help='which prompt to calculate the model for')
    args = parser.parse_args()
    return args.model,args.temperature,args.template,args.runs,args.which,args.prompt

def extract_choice(text, valid=("A", "B")):
    words = str(text).upper().replace(".", " ").split()
    one_letter_words = [w for w in words if w in valid]
    none=('I can not',"I can't", 'I cannot')

    # If response mentions more than one valid option, treat as None
    if any(phrase in text for phrase in none):
        print(text)
        return 'None'
    elif len(set(one_letter_words)) == 1:
        return one_letter_words[0]
    elif len(words)>10:
        #then the reasoning steps are included, thus just take the last token, for behavior gemini
        if (words[-1]=='A')|(words[-1]=='B'):
            return words[-1]
        else:
            return 'None'
    return 'None'


def get_pct_most_exp_option(template,model,temperature='0',mode='csv',df=None,mode_calc='mean'):
    if df.empty:
        df=pd.read_parquet(f'../results/{model}/{template}_{mode}_{model}_{temperature}.parquet')
    run_cols = [col for col in df.columns if col.startswith("output_run")] 
    for col in run_cols:
        df[col + "_choice"] = df[col].apply(lambda x: extract_choice(x, valid=("A","B")))

    choice_cols = [c + "_choice" for c in run_cols]
    if mode_calc=='mean':
        df["exp_count"] = df[choice_cols].eq(df["exp_option"], axis=0).sum(axis=1)
        df["none_count"] = (df[choice_cols] == 'None').sum(axis=1)
        df["exp_percentage"] = df["exp_count"] / len(run_cols)
        df["none_percentage"] = df["none_count"] / len(run_cols)
    elif mode_calc=='std':
        #ToDO implement stddev function for none
        df['std_dev']=df[choice_cols].replace({'A':0,'B':1}).std(axis=1)
    return df

def reshape_for_mixed_logit(template,model,temperature='0',mode='csv',df=None,mode_calc='mean'):
    if df.empty:
        df=pd.read_parquet(f'../results/{model}/{template}_{mode}_{model}_{temperature}.parquet')
    run_cols = [col for col in df.columns if col.startswith("output_run")] 
    print(df.columns)
    for col in run_cols:
        df[col + "_choice"] = df[col].apply(lambda x: extract_choice(x, valid=("A","B")))

    choice_cols = [c + "_choice" for c in run_cols]

    return df

def calculate_acc(model,df,scaler):
    varnames=['view', 'floor', 'access_club', 'free_mini_bar','guest_smartphone','cancellation','price_per_night']
    X_train_scaled = pd.DataFrame(scaler.transform(df[varnames]),columns=df[varnames].columns,index=df[varnames].index)
    pred_choice,probas=model.predict(X=X_train_scaled,varnames=varnames,ids=df['unique_id'], alts=df['Scenario'],return_proba=True)
    # print("\nFirst 5 test observations - features:")
    # print(X_train_scaled[:5])

    # 3. Calculate utilities manually
    # For alternative A (reference, utility = 0)
    V_A = np.zeros(len(X_train_scaled))
    # For alternative B
    V_B = (model.coeff_[0] + 
        model.coeff_[1] * X_train_scaled['view'] +
        model.coeff_[2] * X_train_scaled['floor'] +
        model.coeff_[3] * X_train_scaled['access_club'] +
        model.coeff_[4] * X_train_scaled['free_mini_bar'] +
        model.coeff_[5] * X_train_scaled['guest_smartphone'] +
        model.coeff_[6] * X_train_scaled['cancellation'] +
        model.coeff_[7] * X_train_scaled['price_per_night'])
    
    df_one = df.groupby("unique_id").first().reset_index()
    beta = model.coeff_    # or whatever xlogit stores it as
    print("Converged?", model.convergence)
    print("Total iterations:", model.total_iter)
    print("Log-likelihood:", model.loglikelihood)
    print(model.summary())
    # True labels
    y_true = df_one['output_choice'].values   # should also be 'A'/'B'
    # print(y_true)
    print('aantal antwoorden',len(y_true))

    # Accuracy
    acc = accuracy_score(y_true, pred_choice)
    print("Accuracy:", acc)
    # print(pred_choice,probas)
    # auc=auc()

    return acc

def calculate_r_squared(model,model_null):
    r_squared=1-model.loglikelihood/model_null.loglikelihood
    # print(model.loglikelihood,model_null.loglikelihood)
    return r_squared


def map_values(df_full,HK='yes'):
    df_full['view']=df_full['view'].map({'city':0,'harbour':1})
    df_full['floor']=df_full['floor'].map({'10th':10,'18th':18,'26th':26})
    df_full['access_club']=df_full['access_club'].map({'no':0,'yes':1})
    df_full['free_mini_bar']=df_full['free_mini_bar'].map({'soft drinks, snacks':0,'soft drinks, snacks, wine & beer':1})
    df_full['guest_smartphone']=df_full['guest_smartphone'].map({'not available':0,'available (with free voice + data)':1})
    df_full['cancellation']=df_full['cancellation'].map({'non-refundable':0,'refundable (up to 24 h. prior)':1})
    df_full['price_per_night']=df_full['price_per_night'].map({'HK$ 1600':1600,'HK$ 2000':2000,'HK$ 2400':2400,'HK$ 2800':2800,'HK$ 3200':3200})
    df_full['price_per_night_USD']=df_full['price_per_night_USD'].map({'US$ 312.0':312,'US$ 260.0':260,'US$ 208.0':208,'US$ 364.0':364,'US$ 416.0':416})
    # df_full['output_choice']=df_full['output_choice'].map({'A':0,'B':1})
    return df_full    

def get_multinomial_logit(df,var_names=['view', 'floor', 'access_club', 'free_mini_bar','guest_smartphone','cancellation','price_per_night']):
    #varnames=var_names
    varnames=['view', 'floor', 'access_club', 'free_mini_bar','guest_smartphone','cancellation','price_per_night']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df[varnames])
    #df['price_per_night_in_thousands']=df['price_per_night']/1000
    model = MultinomialLogit()
    # print(df[varnames])
    model.fit(X=X_train_scaled, y=df['output_choice'], varnames=varnames,
            ids=df['unique_id'], alts=df['Scenario'],fit_intercept=True) 
    model_null=MultinomialLogit()
    X_null = np.zeros((len(df), 0))
    model_null.fit(X=X_null, y=df['output_choice'], varnames=[],
            ids=df['unique_id'], alts=df['Scenario'],fit_intercept=True )
    acc=calculate_acc(model,df,scaler)
    
    return model,acc,model_null,scaler


if __name__ == "__main__":
    model,temperature,template,runs,which,prompt = parse_args()
    model_name=model
    mode='csv'
    if (model_name=='gpt-5-mini')|(model_name=='gpt-4o')|(model_name=='gemini-3-pro')|(model_name=='haiku'):
        df=pd.read_parquet(f'results/{model}/full_batch_results_{template}_{model}.parquet')
    else:
        df=pd.read_parquet(f'results/only travel/{model}/{template}_{mode}_{model}_{temperature}_runs_{runs}.parquet')
    if which=='first':
        df = df.iloc[::2]
    elif which=='second':
        df = df.iloc[1::2]

    df_to_merge=pd.read_parquet('vars/hotel_room_dilemmas_shorter_club.parquet')
    df=reshape_for_mixed_logit(template, model,temperature,mode,df=df)
    run_cols = [col for col in df.columns if col.startswith("output_run")]
    choice_cols = [c for c in run_cols if 'choice' in c]
    output_runs_final=set(run_cols).symmetric_difference(choice_cols)
    df_long = df.melt(
        id_vars=[col for col in df.columns if col not in run_cols],  # all other variables
        value_vars=choice_cols,                                         # choice columns only
        var_name="run_variable",
        value_name="output_choice"
    )
    
    #add unique id
    df_long['unique_id']=df_long.index
    df_long=df_long.rename(columns={'scenario_a':'A','scenario_b':'B'})
    df_long=df_long.melt(id_vars=[col for col in df_long.columns if col not in ['A','B']],value_vars=['A','B'],var_name='Scenario', value_name='scenario_prompt')
    df_long=df_long.sort_values(['unique_id','Scenario'])
    df_long=df_long.fillna('')
    print(df_long['scenario_prompt'][0])
    print(df_to_merge['scenario_prompt'][0])
    df_full=pd.merge(df_long,df_to_merge.drop(['dilemma_id'],axis=1).drop_duplicates(),on='scenario_prompt',how='left').dropna()
    print(df_full)
    df_USD_to_merge=df_to_merge.drop(['dilemma_id','scenario_prompt'],axis=1)
    df_full_USD=pd.merge(df_long,df_USD_to_merge.drop_duplicates().rename(columns={'scenario_prompt_USD':'scenario_prompt'}),on='scenario_prompt', how='left').dropna()

    df_full_USD=map_values(df_full_USD,'no')
    df_full=map_values(df_full)

    if prompt=='all':
        amount_info=df_full['amount_info'].unique()
    else:
        amount_info=[prompt]
    subset_df=[]
    subset_df_USD=[]
    for info in amount_info:
        subset_df.append(df_full.loc[df_full['amount_info']==info])
        subset_df_USD.append(df_full_USD.loc[df_full_USD['amount_info']==info])
    
    list_acc=[]
    R_squared=[]
    print(amount_info)
    for df_,info in zip(subset_df,amount_info):
        choice_counts = df_['output_choice'].value_counts()
        min_percentage = choice_counts.min() / len(df_)
        print(min_percentage)
        if (df_['output_choice'].nunique() < 2) | (min_percentage < 0.05):
            print(f"Skipping {info}: Smallest group is {min_percentage:.1%} of data")
            list_acc.append(np.nan)
            R_squared.append(np.nan)
            continue

        model,acc,model_null,scaler=get_multinomial_logit(df_)
        # save the multinomial logit models
        joblib.dump(model, f"results/multinomial_models/{model_name}/multinomial_logit_model_{info}_HK_dilemmas_{which}.joblib")
        joblib.dump(scaler, f"results/multinomial_models/{model_name}/standardscaler/standardscaler_{info}_HK_dilemmas_{which}.joblib")
        r_sqrt=calculate_r_squared(model,model_null)
        list_acc.append(acc)
        R_squared.append(r_sqrt)
        print('r squared HKD: ',r_sqrt)
    
    list_acc_USD=[]
    R_squared_USD=[]
    for df_,info in zip(subset_df_USD,amount_info):
        choice_counts = df_['output_choice'].value_counts()
        min_percentage = choice_counts.min() / len(df_)
        if (df_['output_choice'].nunique() < 2) | (min_percentage < 0.05):
            print(f"Skipping {info}: Smallest group is {min_percentage:.1%} of data")
            list_acc_USD.append(np.nan)
            R_squared_USD.append(np.nan)
            continue
        model,acc,model_null,scaler=get_multinomial_logit(df_)
        # save the multinomial logit models
        joblib.dump(model, f"results/multinomial_models/{model_name}/multinomial_logit_model_{info}_USD_dilemmas_{which}.joblib")
        joblib.dump(scaler, f"results/multinomial_models/{model_name}/standardscaler/standardscaler_{info}_USD_dilemmas_{which}.joblib")
        r_sqrt=calculate_r_squared(model,model_null)
        list_acc_USD.append(acc)
        R_squared_USD.append(r_sqrt)
        print('r squared USD: ',r_sqrt)
    df_acc = pd.DataFrame([list_acc, list_acc_USD],columns=amount_info, index=["HKD", "USD"])
    df_rsqrt = pd.DataFrame([R_squared, R_squared_USD],columns=amount_info, index=["HKD", "USD"])
    df_acc.to_parquet(f"results/multinomial_models/{model_name}/accuracies{which}.parquet")
    df_rsqrt.to_parquet(f"results/multinomial_models/{model_name}/r_sqrts{which}.parquet")