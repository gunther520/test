from distilabel.steps.tasks import UltraFeedback
from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline


with Pipeline(
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:
# Consider this as a placeholder for your actual LLM.
    ultrafeedback = UltraFeedback(
        llm=InferenceEndpointsLLM(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            generation_kwargs={"max_new_tokens": 512},
        ),
        aspect="helpfulness"
    )

    ultrafeedback.load()

    result = next(
        ultrafeedback.process(
            [
                {
                    "instruction": "How much is 2+2?",
                    "generations": ["4", "and a car"],
                }
            ]
        )
    )
    

pipeline.run()