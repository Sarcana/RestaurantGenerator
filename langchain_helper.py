from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key
import os

os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.7)


def generate_name_and_items(cuisine):
    # Chain1: Restaurant Chain
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a  restaurant  which serves {cuisine} suggest  a name for it"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Chain2: Food Item Chain
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for my restaurant {restaurant_name}. Return it as a comma separated list"
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    # Chain 3: Sequential Chain
    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )
    response = chain({'cuisine': cuisine})

    return response


if __name__ == '__main__':
    print(generate_name_and_items("Italian"))
