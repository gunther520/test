

from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromHub, KeepColumns, StepResources, step, StepInput, StepOutput
from distilabel.steps.tasks import TextGeneration, UltraFeedback
import numpy as np


# Normal step
@step(inputs=["generations", "ratings"], outputs=["generation"])
def GenerationStep(inputs: StepInput) -> StepOutput:
    for input in inputs:
        input["generation"] = input["generations"][np.argmax(input["ratings"])]
    yield inputs


with Pipeline("pipe-name-2", description="My first pipe2s") as pipeline:
    load_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    combine_generations = GroupColumns(
        name="combine_generations",
        columns=["generation", "model_name"],
        output_columns=["generations", "generation_models"],
    )

    ultrafeedback = UltraFeedback(
        use_default_structured_output=True,
        name="ultrafeedback_1",
        resources=StepResources(gpus=2,replicas=1),
        llm=vLLM(model="meta-llama/Llama-3.1-8B-Instruct",
                 extra_kwargs={"tensor_parallel_size": 2,"distributed_executor_backend": "ray",},
                    ),
        input_mappings={"instruction": "instruction", "generations": "generations"},
        aspect="overall-rating",
        output_mappings={"model_name": "ultrafeedback_model"},
    ) 

    keep_columns = KeepColumns(
        name="keep_columns",
        columns=[
            "instruction",
            "generations",
            "generation_models",
            "ratings",
            "rationales",
            "ultrafeedback_model",
        ],
    )




    for llm in (
        vLLM(model="meta-llama/Llama-3.2-1B-Instruct",
            #cuda_devices=[0],
            extra_kwargs={"distributed_executor_backend": "ray",}),
        vLLM(model="meta-llama/Llama-3.2-1B",
            #cuda_devices=[1],
            extra_kwargs={"distributed_executor_backend": "ray",}),
    ):
        task = TextGeneration(
            name=f"text_generation_with_{llm.model_name[-2:]}", llm=llm
        )
        load_dataset.connect(task)
        task.connect(combine_generations)
    combine_generations.connect(ultrafeedback)
    ultrafeedback.connect(keep_columns)
    keep_columns.connect(GenerationStep(name="generation_step"))

if __name__ == "__main__":
    distiset = pipeline.run(
        use_cache=False,
        parameters={
            "load_dataset": {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            "text_generation_with_ct": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                },
                "resources": { "gpus": 1}
            },
            "text_generation_with_1B": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                },
                "resources": { "gpus": 1}
            },

            "ultrafeedback_1": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 1024,
                    }
                },
            },
        }
    )

    distiset.push_to_hub(
        "Gunther520/instruction-dataset-mini-with-generations"
    )