import os
from pathlib import Path
from ..app_types import ProviderRequest
from .base import Provider
from openai import OpenAI
import json
ROOT = Path(__file__).resolve().parents[2]

class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.key,)

    def batch_complete(self, req: ProviderRequest,template,runs) ->str:
        api='openai'
        for ind,conv in enumerate(req.messages):
            input_mes = [{"role": msg.role, "content": msg.content} for msg in conv] 
            batch_input=[{ "custom_id": f"request-{ind}-{run}", "method": "POST", "url": "/v1/chat/completions", "body": { "model": req.model, "messages": input_mes ,"temperature":req.temperature}} for run in range(0,runs)]
            if ind %10 ==0:
                full_batch_input_list=[]
                corr_ind_file=ind
                with open(ROOT/"batches"/req.model/f"{template}"/f"batchinput_{api}_{ind}.jsonl","w") as f:
                    for d in batch_input:
                        json.dump(d,f)
                        f.write("\n")
                        full_batch_input_list.append(d)
            else:
                with open(ROOT/"batches"/req.model/f"{template}"/f"batchinput_{api}_{corr_ind_file}.jsonl", "a") as f:
                    for d in batch_input:
                        json.dump(d,f)
                        f.write("\n")
                        full_batch_input_list.append(d)
            if ((ind+1)%10==0) or (ind+1==len(req.messages)):
                response = self.client.files.create(file=open(ROOT/"batches"/req.model/f"{template}"/f"batchinput_{api}_{corr_ind_file}.jsonl",'rb'), purpose='batch' )
                file_id = response.id 
                response = self.client.batches.create( input_file_id=file_id, endpoint='/v1/chat/completions', completion_window='24h' ) 
                batch_id = response.id
                with open(ROOT/"batches"/req.model/f"{template}"/f"{api}_{template}_batchids.txt", "a") as f:
                    f.write(batch_id)  # add batch_id
                    f.write("\n")

        return full_batch_input_list
