# llm router project

The goal of this project is to create an LLM router.
This first version of the router is able to dynamically route between an inferior and a superior model according to a user prompt. 
The benefit of the router is that it can select the inferior model, when the prompt is easy enough and thus, without sacrificing accuracy, it can optimise for cost.

## project structure

The code of the repo is structured in the following steps: 
* `dataset_analysis`: the script of this folder is used to analyse the `lmsys/mt_bench_human_judgments` dataset. This dataset consists of pairwise human preferences for model responses. We assume a categorisation of models to inferior/superior, based on this list. Then, we rank the difficutly of the questions based on the human preference ratio of inferior vs superior. This way, we can label the questions according to the preferred route destination. Using the questions, we can create a carefully crafted prompt that will be used in the next step.
* `create_dataset`: this script uses the carefully crafted prompt to create a synthetic dataset. More specifically, we use the `lmsys/lmsys-chat-1m` dataset and label the prompts according to preferred route destination. I have saved a combination of the synthetic data I have collected under the `final-datasets` directory.
* `finetuning_model`: after having produced enough synthetic data, we use this script to fine-tune BERT in order to classify prompts. We save the `model` in order to use it later. We also measure the performance of the finetuned model against the original questions dataset (which can be found labelled under the `final-datasets` directory). This file is supposed to be run in Google Colab with use of GPUs.
* `inference`: lastly, using the finetuned pretrained model, we create this inference UI page, where users can type different prompts and check the routing prediction of the finetuned model.

The `final-datasets`, `models` directory and the `.env` that contains private tokens are not pushed to the public directory.

## how to run

### virtualenv

```bash
## to create the virtual env
python3 -m pip install virtualenv   # install if not already in the system
virtualenv router-venv
source router-venv/bin/activate

## to create requirements automatically
python3 -m pip install pipreqs
pipreqs . --force --ignore bin,etc,include,lib,lib64    # avoiding problems with venv
pip freeze > requirements.txt   # another way to create requirements.txt

## to use it
source router-venv/bin/activate
python3 -m pip install -r requirements.txt
# assuming you have your pretrained model ready under `models/` directory
python3 inference/run_gradio_model.py --model_save_dir ./models/bert-base-uncased-router-finetuning-20240722T133228-save --hf_model_name bert-base-uncased
deactivate
```

In this demo app, you can type any request you want and the router model will display the probability of routing to the superior vs the inferior LLM.
In general, the more specific / unusual / difficult the request, the higher the probability of routing to the superior model.

### terminal commands

Merging multiple files into one: 
* `cat datasets/router_dataset_labelled-openai_gpt-3.5-turbo-*.jsonl > router_dataset_all.jsonl`

Keeping only inferior/superior tags dataset:
* `grep '"ROUTE_TO_INFERIOR"' router_dataset_all.jsonl > router_dataset_inferior.jsonl`
* `grep '"ROUTE_TO_SUPERIOR"' router_dataset_all.jsonl > router_dataset_superior.jsonl`
