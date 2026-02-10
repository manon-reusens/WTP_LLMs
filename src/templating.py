from pathlib import Path
from typing import Dict
import yaml
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
import random

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
VARS_DIR = ROOT / "vars"

env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    undefined=StrictUndefined,  # fail fast if a var is missing
    trim_blocks=True,
    lstrip_blocks=True,
)

def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_vars(template,source,csv_path,model,temperature,yaml_file) -> list:
    """
    the source from where the data is loaded is chosen here, either csv or yaml.
    """
    defaults = load_yaml(VARS_DIR / yaml_file)
    dollars=defaults['dollar']
    dilemmas=defaults['dilemma']
    prev_info=defaults['previous_info']
    variables=[]
    if template=='hotel_room':
        country='none'
        df=pd.read_parquet('vars/hotel_room_dilemmas.parquet')
        for i in range(0,int(len(df)/2)):
            for dtype3,info in prev_info['hotel_room'].items():
                subset_df=df.loc[df['dilemma_id']==i]
                all_variables={
                                "nationality": country,
                                "application":'hotel_room',
                                "dilemma_type": i,
                                "scenario_a": subset_df['scenario_prompt'].iloc[0],
                                "scenario_b": subset_df['scenario_prompt'].iloc[1],
                                "amount_info":dtype3,
                                "prev_info":info
                            }
                variables.append(all_variables)
                all_variables={
                                "nationality": country,
                                "application":'hotel_room',
                                "dilemma_type": i,
                                "scenario_a": subset_df['scenario_prompt'].iloc[1],
                                "scenario_b": subset_df['scenario_prompt'].iloc[0],
                                "amount_info":dtype3,
                                "prev_info":info
                            }
                variables.append(all_variables)
                all_variables={
                                "nationality": country,
                                "application":'hotel_room',
                                "dilemma_type": i,
                                "scenario_a": subset_df['scenario_prompt_USD'].iloc[0],
                                "scenario_b": subset_df['scenario_prompt_USD'].iloc[1],
                                "amount_info":dtype3,
                                "prev_info":info
                            }
                variables.append(all_variables)
                all_variables={
                                "nationality": country,
                                "application":'hotel_room',
                                "dilemma_type": i,
                                "scenario_a": subset_df['scenario_prompt_USD'].iloc[1],
                                "scenario_b": subset_df['scenario_prompt_USD'].iloc[0],
                                "amount_info":dtype3,
                                "prev_info":info
                            }
                variables.append(all_variables)
    else:
        print('Other templates are not supported')
                    
    return variables

def render_system() -> str:
    return (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()

def render_user(template_name: str, variables: list) -> str:
    # templates live under prompts/user/*.j2
    template = env.get_template(f"user/general_nonationality.j2")
    outputs=[]
    for row in variables:
        first_pass = template.render(**row).strip()
        # second_pass = Template(first_pass).render(**row)
        # outputs.append(second_pass)
        outputs.append(first_pass)
    return outputs
