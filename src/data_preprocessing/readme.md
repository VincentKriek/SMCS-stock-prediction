# Data pre-processing

## Preparation

In the project root create the `data/pre-processor` folder. Also, the news and prices file from the [data loader step](../data_loader/readme.md) should be in the `data/loader` folder, where the timestamps in the files are matching. For the data pre-processing step the Qwen2.5:7B model is used. This model has to be locally run through ollama. Follow [the ollama installation instructions](https://ollama.com/download/windows) to install an run ollama on your machine.

## Running the code

The code will generate the LLM sentiment scores and finally create the data to run the models. The code can be run using

```sh
$ uv run src/data_preprocessing/main.py
```

Since generating the sentiment scores takes a long time, all scores are stored in a `checkpoint_{{min_date}}_{{max_date}}.jsonl` file. The code can be stopped and continued at any time, without losing the intermediate work. When all scores are generated, the model will format it in a way so it can be handled by the model. This will result in a file of the following shape

```
prepared_data_{{min_date}}_{{max_date}}.parquet
```