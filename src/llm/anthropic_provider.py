import os
from pathlib import Path
from ..app_types import ProviderRequest
from .base import Provider
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import json
ROOT = Path(__file__).resolve().parents[2]

class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self):
        self.key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.key,)

    def batch_complete(self, req: ProviderRequest,template,runs) ->str:
        api='anthropic'
        corr_ind_file = 0
        for ind,conv in enumerate(req.messages):
            input_mes = [{"role": msg.role, "content": msg.content} for msg in conv] 
            system_prompt=input_mes[0]['content']
            input_mes=input_mes[1:]
            batch_input=[{ "custom_id": f"request-{ind}-{run}", "params": { "model": req.model, "messages": input_mes ,"system":system_prompt,"temperature":req.temperature,'max_tokens':req.max_tokens if hasattr(req, 'max_tokens') else 1024,}} for run in range(0,runs)]
            if ind %300 ==0:
                full_batch_input_list=[]
                batch_requests = []
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
            for run in range(0, runs):
                batch_requests.append(
                    Request(
                        custom_id=f"request-{ind}-{run}",
                        params=MessageCreateParamsNonStreaming(
                            model=req.model,
                            max_tokens=req.max_tokens if hasattr(req, 'max_tokens') else 1024,
                            messages=input_mes,
                            temperature=req.temperature,
                            system=system_prompt
                        )
                    )
                )
            if ((ind+1)%300==0) or (ind+1==len(req.messages)):
                message_batch = self.client.messages.batches.create(
                    requests=batch_requests
                )
                
                batch_id = message_batch.id
                with open(ROOT/"batches"/req.model/f"{template}"/f"{api}_{template}_batchids.txt", "a") as f:
                    f.write(batch_id)  # add batch_id
                    f.write("\n")

                batch_requests = []
        return full_batch_input_list
