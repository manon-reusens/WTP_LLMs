import argparse
from pathlib import Path
import yaml
from .templating import load_vars, render_system, render_user
from .router import get_provider, resolve_model, default_config
from .app_types import Message, ProviderRequest
import pandas as pd
from tqdm import tqdm
import json

def run(template, provider, model, temperature, vars_source, csv_path,runs,num_gpus,yaml_file="default.yaml"):
    def_provider, def_model, def_temp= default_config()
    # pull default from config if present
    try:
        from pathlib import Path
        ROOT_PATH=Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((ROOT_PATH / "config" / "app.yaml").read_text())
        cfg_vars_source = cfg.get("default_vars_source", "yaml") #dictionary look up with default yaml
    except Exception:
        cfg_vars_source = "yaml"
    model_alias=model
    provider = def_provider if provider is None else provider
    model = resolve_model(provider, model or def_model)
    temperature = def_temp if temperature is None else temperature
    vars_source = cfg_vars_source if vars_source is None else vars_source

    variables = load_vars(template=template,
        source=vars_source,                      
        csv_path=Path(csv_path) if csv_path else None,model=model_alias,temperature=temperature,yaml_file=yaml_file
    )
    system = render_system()
    user = render_user(template, variables)
    if template=='general_nonationality':
        llm_hist=[var['history_llm']for var in variables]
        user_hist=[var['history_user']for var in variables]

    client = get_provider(provider)
    if (provider=='openai')|(provider=='google')|(provider=='anthropic'):
        if template=='general_nonationality':
            req= ProviderRequest(
                    model=model,
                    messages=[[Message(role="system", content=system),Message(role="user", content=u1),Message(role="assistant", content=a1),Message(role="user", content=u)]for u1,a1,u in zip(user_hist,llm_hist,user)],
                    temperature=temperature,)
        else:
            req= ProviderRequest(
                        model=model,
                        messages=[[Message(role="system", content=system),Message(role="user", content=u)]for u in user],
                        temperature=temperature,)
        inputs=client.batch_complete(req,template,runs) 
        for var, inp in zip(variables, inputs):
            var["input"] = inp

    elif provider=='huggingface':
        #the req is the same as previously, oly the messages are a list of 
        if template=='general_nonationality':
            print([[Message(role="system", content=system),Message(role="user", content=u1),Message(role="assistant", content=a1),Message(role="user", content=u)]for u1,a1,u in zip(user_hist,llm_hist,user)])
            req= ProviderRequest(
                    model=model,
                    messages=[[Message(role="system", content=system),Message(role="user", content=u1),Message(role="assistant", content=a1),Message(role="user", content=u)]for u1,a1,u in zip(user_hist,llm_hist,user)],
                    temperature=temperature,)
        else:
            req= ProviderRequest(
                        model=model,
                        messages=[[Message(role="system", content=system),Message(role="user", content=u)]for u in user],
                        temperature=temperature,)
        outputs,inputs=client.batch_complete(req,runs,num_gpus) 
        if not (len(variables) == len(outputs) == len(inputs)):
            raise ValueError("variables, outputs, and inputs must all be the same length")
        print(inputs)
        for var, out_runs, inp in zip(variables, outputs, inputs):
            for r, out in enumerate(out_runs):
                var[f"output_run{r}"] = out
                var["input"] = json.dumps(inp)
        
    final_output=pd.DataFrame(variables)
    final_output.to_parquet(ROOT_PATH/"results"/model_alias/f"{template}_{vars_source}_{model_alias}_{temperature}_runs_{runs}.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run an LLM call with chosen template and vars source.")
    ap.add_argument("--template", required=True, help="Template name without .j2, e.g. flight")
    ap.add_argument("--vars-source", choices=["csv","csv_country","csv_currency", "yaml"], help="Where variables come from")
    ap.add_argument("--csv-path", help="Override path to nationalities.csv")
    ap.add_argument("--provider", help="openai|groq|anthropic|huggingface")
    ap.add_argument("--model", help="Alias from providers.yaml or raw id")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--runs",type=int,default=1, help="the number of runs for this model")
    ap.add_argument("--num_gpus",type=int,default=1,help="the number of GPUs to run this experiment on")
    ap.add_argument("--yaml_file",type=str,default="default.yaml",help="the yaml_file to extract the dilemmas from")
    args = ap.parse_args()
    run(args.template,args.provider, args.model, args.temperature, args.vars_source, args.csv_path,args.runs,args.num_gpus,args.yaml_file)
