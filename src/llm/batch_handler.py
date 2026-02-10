import argparse
from openai import OpenAI
import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from google import genai
import anthropic
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
print(ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Gather the results from the batches.")
    parser.add_argument('--batch_ids_path', type=str, required=True, help="path to file containing the batch ids")
    parser.add_argument('--output_path', type=str, required=True, help="path to where you want to save the output file")
    parser.add_argument('--combine_path', type=str,default='',help="the path to the file you want to combine the batch results with") 
    parser.add_argument('--prompt',type=str,choices=['hotel_room','flight','hotel','transportation','general_template','general_nonationality','general_nonationality_no_history','generate_history'],help='the prompt you are evaluating')
    parser.add_argument('--model',type=str,choices=['gpt-4o-mini','gpt-4o','gpt-5-mini','gpt-5','haiku','gemini-3-pro'],help='the model used')
    args = parser.parse_args()
    return args.batch_ids_path,args.output_path,args.combine_path,args.prompt,args.model


def get_batch_results_openai(batch_id,df):
    #start connection
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key,)
    #retrieve the file output using the batch id
    batch = client.batches.retrieve(batch_id)
    # print(batch)
    print(batch)
    file_id=batch.output_file_id
    file_response = client.files.content(file_id)
    #splitlines, as the output is a jsonlines file
    lines = file_response.text.splitlines()
    data = [json.loads(line) for line in lines]
    #gather the content, batch_ids, and custom_ids
    content=[data_point['response']['body']['choices'][0]['message']['content'] for data_point in data]
    batch_ids=[batch_id for data_point in data]
    custom_ids=[data_point['custom_id']for data_point in data]
    full_data={'batch_id':batch_ids,'custom_id':custom_ids,'content':content}
    #add to dataframe and combine both
    df_new=pd.DataFrame(full_data)
    df=pd.concat([df,df_new],ignore_index=True)
    return df

def get_batch_results_claude(batch_id,df):
    #start connection
    key = os.getenv("ANTHROPIC_API_KEY")
    client=anthropic.Anthropic(api_key=key)
    #retrieve the file output using the batch id
    content=[]
    batch_ids=[]
    custom_ids=[]
    for result in client.messages.batches.results(batch_id):
        content.append(result.result.message.content[0].text)
        batch_ids.append(batch_id)
        custom_ids.append(result.custom_id)
    full_data={'batch_id':batch_ids,'custom_id':custom_ids,'content':content}
    #add to dataframe and combine both
    df_new=pd.DataFrame(full_data)
    df=pd.concat([df,df_new],ignore_index=True)
    return df

def get_batch_results_google(batch_id,df):
    #start connection
    key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key,)
    #retrieve the file output using the batch id
    batch = client.batches.get(name=batch_id)
    # print(batch)
    print(batch)
    file_id=batch.dest.file_name
    file_response = client.files.download(file=file_id)
    # return file_response
    #splitlines, as the output is a jsonlines file
    text =file_response.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    data = [json.loads(line) for line in lines]
    rows = []
    for obj in data:
        try:
            text_out = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue  # skip entries without parts

        rows.append({
            "batch_id": batch_id,
            "custom_id": obj.get("key"),
            "content": text_out
        })

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

if __name__ == "__main__":
    batch_id_path,output,combine_path,prompt,model = parse_args()
    runs=1
    with open(batch_id_path,'r') as f:
        data=f.readlines()
    if 'anthropic' in batch_id_path:
        eval_api='claude'
    elif 'openai' in batch_id_path:
        eval_api='openai'
    elif 'google' in batch_id_path:
        eval_api='google'
    else:
        print('Only Claude and Openai are supported')
    df=pd.DataFrame(columns=['batch_id','custom_id','content'])
    for batch_id in data:
        batch_id_final=batch_id.replace('\n','')
        if (eval_api=='') or (eval_api=='openai'):
            df=get_batch_results_openai(batch_id_final,df)
        elif eval_api=='google':
            df=get_batch_results_google(batch_id_final,df)
        elif (eval_api=='claude'):
            df=get_batch_results_claude(batch_id_final,df)
    df.to_parquet(output+f'/{model}/batch_results_{prompt}_{model}.parquet')
    
    if combine_path!='':
        df_original=pd.read_parquet(combine_path)
        for r in range(0,runs):
            df_original[f'output_run{r}']=np.nan
        for i, row in df_original.iterrows():
            for r in range(0,runs):
                df_original.at[i,f'output_run{r}']=df.loc[df['custom_id']==f"request-{i}-{r}", 'content'].values[0]
        # df_original['batch_id']=np.nan
        # for r in range(0,runs):
        #     df_original[f'output_run{r}']=np.nan
        # print('this should be checked for multiple questions, but I think ti will be twice the same number')
        # for index,row in df_original.iterrows():
        #     print('might want to optimize this so that you can ask all 3 prompts at the same time?')
        #     df_original.at[index,'batch_id']=batch_id_final
        #     for r in range(0,runs):
        #         custom_id=f'request-{str(index)}-{str(r)}'
        #         df_original.at[index,f'output_run{r}']=df.loc[(df['custom_id']==custom_id)]['content'].iloc[0]
        df_original.to_parquet(output+f'/{model}/full_batch_results_{prompt}_{model}.parquet')
