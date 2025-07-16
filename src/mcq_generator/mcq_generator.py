from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv()


key = os.getenv("OPENAI_API_KEY")
api_base_url = os.getenv("OPENAI_API_BASE")

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo-16k",
    temperature=0.7,
    openai_api_key=key,
    openai_api_base=api_base_url,
    
)



quiz_generation_template = PromptTemplate(
    input_variables=["text", "number", "subject", "response_json"],
    template=TEMPLATE
)

quiz_evaluation_template = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)


chain1 = LLMChain(llm=llm, prompt=quiz_generation_template, output_key="quiz", verbose=True)
chain2 = LLMChain(llm=llm, prompt=quiz_evaluation_template, output_key="review", verbose=True)

quiz_generation_chain = SequentialChain(chains=[chain1, chain2], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True)


