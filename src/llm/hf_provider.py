import os
from ..app_types import ProviderRequest
from .base import Provider
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import login
import json
# cache_dir='CACHE_DIR'
# os.environ['TRANSFORMERS_CACHE'] =cache_dir


class HFProvider(Provider):
    name = "huggingface"

    def __init__(self):
        self.key = os.getenv("HUGGINGFACE_API_KEY")
        self.cache_dir=cache_dir
        login(self.key)
    def batch_complete(self, req: ProviderRequest,runs,num_gpus) ->str:
        response,input_list=self.vllm_generate(input_list = [[{"role": msg.role, "content": msg.content} 
                                                              for msg in conversation] for conversation in req.messages],
                                                              model_name=req.model,temperature=req.temperature,runs=runs,num_gpus=num_gpus)
        print(response)
        print('in batch complete the input_list')
        print(input_list)
        return response,input_list


    def  vllm_generate(self, input_list,model_name,temperature,runs=1,seed=0,num_gpus=1):
        #note: P100 and V100 not supported
        NUM_GPUS = num_gpus
        save_path=f"intermediate_results_{model_name.replace('/','')}_{temperature}.jsonl"
        # Render input text into llm format
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.key)
        model = LLM(model=model_name, tensor_parallel_size=NUM_GPUS,seed=seed,dtype='bfloat16',max_model_len=2048,download_dir=self.cache_dir,gpu_memory_utilization=0.95)
        # Model generation
        #sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=1024) #as long as temp=0 then top_p does not matter
        batch_size=32
        output_list = []

        # If resuming, load existing results
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing = [json.loads(line) for line in f]
            done_inputs={json.dumps(item["input"], sort_keys=True) for item in existing}
            output_list.extend(existing)
        else:
            done_inputs = set()
        print(done_inputs)

        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i+batch_size]
            # skip inputs already processed
            #batch = [x for x in batch if x not in done_inputs]
            if not batch:
                continue
            #print('Batch:',batch)
            batch_tok = [tokenizer.apply_chat_template(user_input, tokenize=False, add_special_tokens=False, add_generation_prompt=True) for user_input in batch]
            # collect multiple runs per input
            run_outputs = [[] for _ in batch]  # one slot per input

            for run in range(runs):
                sampling_params = SamplingParams(temperature=temperature,seed=seed+run, top_p=0.95, max_tokens=1024) #added it here for reproducibility to have one seed per run, only important if temp not 0
                batch_output = model.generate(batch_tok, sampling_params=sampling_params, use_tqdm=True)
                for j, output in enumerate(batch_output):
                    run_outputs[j].append(output.outputs[0].text.strip())
                    

            #output_list.extend([output.outputs[0].text.strip() for output in batch_output])
            output_list.extend(run_outputs)
            for inp, outs in zip(batch, run_outputs):
                record = {"input": inp, "outputs": outs}
                #output_list.append(record)
                with open(save_path, "a") as f:  # append mode
                    f.write(json.dumps(record) + "\n")

        return output_list,input_list
