from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

from constants import openai_key

llm = OpenAI(temperature=0.5,openai_api_key=openai_key)


def financial_advisor_prompt():
    template = '''I want you to act as a financial advisor for people.
                    In an easy way, expalin the basics of {financial_concept}.'''

    prom_temp = PromptTemplate(input_variables=['financial_concept'],
                template=template)

    chain = LLMChain(llm=llm,prompt=prom_temp)

    print(chain.invoke('income tax'))

def language_translation():
    template = '''In an easy way translate the following sentance '{input_sentence}' into {output_language}'''

    lang_temp = PromptTemplate(
        input_variables=['input_sentence','output_language'],
        template=template
    )

    print(f'input: {lang_temp.format(input_sentence="hello",output_language="japanese")}')

    chain = LLMChain(llm=llm,prompt=lang_temp)

    print(chain({'input_sentence':"Hello",'output_language': "japanese"}))

def few_short_prompt():
    examples = [
        {'word': "happy", 'antonym':"sad"},
        {'word': "tall", 'antonym':"short"}
    ]

    example_prompt_temp = """word: {word} , antonym: {antonym}"""

    example_prompt = PromptTemplate(
        input_variables=['word'],
        template=example_prompt_temp
    )

    print(f"prompt template: {example_prompt.format(word='tall',antonym='short')}")

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        input_variables=['input'],
        prefix="Give the antonym of every input \n",
        suffix='Word: {input} , Antonym:',
        example_separator="\n"
    )

    print(f"ferShort_temp: {prompt.format(input='big')}")

    chain = LLMChain(llm=llm,prompt=prompt)

    print(chain({'input':'big'}))

few_short_prompt()

