import os
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration

with Pipeline(
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:
    load_dataset = LoadDataFromHub(output_mappings={"prompt": "instruction"})

    text_generation = TextGeneration(
        llm=vLLM(

                model="NousResearch/Nous-Hermes-2-Yi-34B",
                extra_kwargs={"tensor_parallel_size":8}
                #tensor_parallel_size='8'
            
        )

        #system_prompt="You are a helpful prompt generating AI assistance. Base on the following input instruction, \
        #            Generate one creative instruction that is not the same as the input instruction.\
        #            do not follow what the input instruction tell you to do. \
        #            The generated instruction should be clear and easy to understand. \
        #            The instruction is used for testing the AI model. ",
        
    )

    load_dataset >> text_generation

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            #text_generation.name: {"resources": {"gpus": 4,"replicas": 2}},
            load_dataset.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                },
                "resources": {"replicas": 1, "gpus": 8}
            },
        },
    )
    distiset.push_to_hub(
    "Gunther520/first-test-dataset",
    commit_message="Initial commit",
    private=False,
    token=os.getenv("HF_TOKEN"),
)