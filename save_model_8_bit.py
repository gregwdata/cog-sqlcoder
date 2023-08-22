import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    return prompt


def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        #torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
        load_in_8bit=True
    )
    return tokenizer, model

def run_inference(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    tokenizer, model = get_tokenizer_model("defog/sqlcoder")
#     prompt = generate_prompt(question, prompt_file, metadata_file)
    
#     # make sure the model stops generating at triple ticks
#     eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=1200,
#         do_sample=False,
#         num_beams=5, # do beam search with 5 beams for high quality results
#     )
#     #pipe.save_pretrained('./sqlcoder_8bit')
    model.save_pretrained('./sqlcoder_8bit/model')
    tokenizer.save_pretrained('./sqlcoder_8bit/tokenizer')
    print('model saved')
    return None

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run inference on a question")
    parser.add_argument("-q","--question", type=str, help="Question to run inference on")
    args = parser.parse_args()
    question = args.question
    print("Loading a model and generating a SQL query for answering your question...")
    print(run_inference(question))