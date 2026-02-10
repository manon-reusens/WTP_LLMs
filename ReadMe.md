# Would a Large Language Model Pay Extra for a View? Inferring Willingness to Pay from Subjective Choices
This is the repository belonging to the paper <i>>Would a Large Language Model Pay Extra for a View? Inferring Willingness to Pay from Subjective Choices'</i>



## Installation
```
$ conda create --name wtp_llms
$ conda activate wtp_llms
$ pip install -r requirements.txt
```
## Experiments

All experiments are run with the following python command. All different models should be given in separate runs, together with the correct provider. Below you can find an example of how to start experiments for LLama3B.
```
$ python src/main.py --template 'hotel_room' \
$                 --provider 'huggingface' \
$                 --model "llama-3.2-3B" \
$                 --temperature 0 \
$                 --runs 1 \
$                 --yaml_file 'default_only_travel.yaml' \
```
## Multinomial Logit Models
To subsequently make the Multinomial Logit models from the LLM responses, the following python command should be used. You should choose whether you want only the first, second, or both results of the tests with swapped scenarios after the 'which' command. If you only want to get the model for a specific prompt, specify this, otherwise you can set the value to 'all' and all multionmial logit models will be generated.
```
$ python analysis_results/get_multinomial_logit_models.py --model 'llama-3.2-3B' \
$                                                         --temperature '0.0' \
$                                                         --template 'hotelroom'
$                                                         --runs '1' \
$                                                         --which 'first' \
$                                                         --prompt 'all' \
```
## Calculate WTP
To calculate the WTP values, you use the following commands after the multinomial logit models were made. Here you can also specify for which currency the experiments were run (HK/USD).
```
$ python analysis_results/calculate_WTP.py --model 'llama-3.2-3B' \
$                                          --which 'first' \
$                                          --prompt 'all' \
$                                          --currency 'HK' \
```