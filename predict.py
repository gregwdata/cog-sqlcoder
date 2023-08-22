# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

        
import time
from typing import Optional
import subprocess

import torch
import os

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from collections import OrderedDict
from cog import BasePredictor, ConcatenateIterator, Input, Path

DEFAULT_CONFIG_PATH = "model/"
TOKENIZER_PATH = "model/"

def maybe_download(path):
    if path.startswith("gs://"):
        st = time.time()
        output_path = "/tmp/weights.tensors"
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        print(f"weights downloaded in {time.time() - st}")
        return output_path
    return path

def generate_prompt(question, prompt_string=None, metadata_string=None, prompt_file="prompt.md", metadata_file="metadata.sql"):
    
    if prompt_string:
        prompt = prompt_string
    else:
        with open(prompt_file, "r") as f:
            prompt = f.read()

    if metadata_string:
        table_metadata_string = metadata_string
    else:
        with open(metadata_file, "r") as f:
            table_metadata_string = f.read()

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    return prompt


class Predictor(BasePredictor):
    def get_tokenizer_model(self, model_path, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            #torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
            #load_in_8bit=True
        )
        return tokenizer, model


    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set TOKENIZERS_PARALLELISM to false to avoid a warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # self.model = self.load_tensorizer(
        #     weights=maybe_download(TENSORIZER_WEIGHTS_PATH), plaid_mode=True, cls=YieldingReplitCode, config_path=DEFAULT_CONFIG_PATH,
        # )
        self.tokenizer, self.model = self.get_tokenizer_model("./sqlcoder_8bit/model/","./sqlcoder_8bit/tokenizer/")
        
        print('setup complete')
    


    def predict(
        self,
        prompt: str = Input(description=f"Text prompt"),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=1200,
        ),
        num_beams: int = Input(
            description="Number of beams for beam search",
            ge=1,
            default=5,
        ),
        prompt_template: str = Input(
            description="Prompt template for SQLcoder - see https://github.com/defog-ai/sqlcoder/blob/main/prompt.md . Must contain {user_question} and {table_metadata_string} parameters.",
            default=r"""### Instructions:
Your task is convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{user_question}`.
This query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{user_question}`:
```sql""",
        ),
        schema_metadata: str = Input(
            description="Description of database schema. See https://github.com/defog-ai/sqlcoder/blob/main/metadata.sql ",
            default=r"""CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson 
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region 
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred 
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id 
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id
""",
        ),
        seed: int = Input(
            description="Set seed for reproducible outputs. Set to -1 for random seed.",
            ge=-1,
            default=-1,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator[str]:
        prompt = generate_prompt(prompt,prompt_string=prompt_template,metadata_string=schema_metadata)
        #input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # set torch seed
        if seed == -1:
            torch.seed()

        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
                
        eos_token_id = self.tokenizer.convert_tokens_to_ids(["```"])[0]
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_length,
            do_sample=False,
            num_beams=5, # do beam search with 5 beams for high quality results
        )
        generated_query = (
            pipe(
                prompt,
                num_return_sequences=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )[0]["generated_text"]
            .split("```sql")[-1]
            .split("```")[0]
            .split(";")[0]
            .strip()
            + ";"
        )        

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
            
        #print(generated_query)
            
        return generated_query