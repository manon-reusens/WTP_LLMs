import os
from pathlib import Path
from ..app_types import ProviderRequest
from .base import Provider
from google import genai
from google.genai import types
import json
ROOT = Path(__file__).resolve().parents[2]

class GoogleProvider(Provider):
    name = "google"

    def __init__(self):
        self.key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.key,)

    def batch_complete(self, req: ProviderRequest,template,runs) ->str:
        api='google'
        for ind,conv in enumerate(req.messages):
            input_mes = [{"parts": [{'text':msg.content}], "role": msg.role} for msg in conv]  #bekijken!
            system_prompt=input_mes[0]
            input_mes=input_mes[1:]
            batch_input=[{ "key": f"request-{ind}-{run}", "request": { "contents": input_mes ,"systemInstruction":system_prompt,"generation_config":{"temperature":req.temperature}}} for run in range(0,runs)]
            if ind>13799:
                if ind %300 ==0:
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
                if ((ind+1)%300==0) or (ind+1==len(req.messages)):
                    uploaded_file = self.client.files.upload(file=open(ROOT/"batches"/req.model/f"{template}"/f"batchinput_{api}_{corr_ind_file}.jsonl",'rb'), config=types.UploadFileConfig(display_name=f"batchinput_{api}_{corr_ind_file}", mime_type='jsonl'))
                    file_name = uploaded_file.name 
                    file_batch_job = self.client.batches.create(model=req.model, src=file_name, config={'display_name':f'file_upload_job'}) 
                    batch_id = file_batch_job.name
                    with open(ROOT/"batches"/req.model/f"{template}"/f"{api}_{template}_batchids.txt", "a") as f:
                        f.write(batch_id)  # add batch_id
                        f.write("\n")

        return full_batch_input_list
