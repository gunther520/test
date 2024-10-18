import os
from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration



with Pipeline(
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:
    load_dataset = LoadDataFromHub(output_mappings={"prompt": "instruction"})

    text_generation = TextGeneration(
        llm=InferenceEndpointsLLM(
            #model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            
            model_id="mistralai/Mistral-Nemo-Instruct-2407"
            #tokenizer_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        ),
        system_prompt="You are a creative AI Assistant writer. Base on the input, \
                generate only one similar instruction using one but differnt approach, not the same!! creative instruction for evaluation of the LLM.",
        
    )

    load_dataset >> text_generation

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.3,
                        "max_new_tokens": 512,
                    }
                }
            },
        },
    )
    distiset.push_to_hub(
    "Gunther520/first-test-dataset",
    commit_message="Initial commit",
    private=False,
    token=os.getenv("HF_TOKEN"),
)