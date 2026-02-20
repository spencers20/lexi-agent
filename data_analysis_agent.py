#import the libraries required
from langchain_core.messages import HumanMessage,AIMessage
from typing import TypedDict,Annotated
from langchain_openai import AzureChatOpenAI
import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO,StringIO
import base64
import sys
from pydantic import BaseModel, Field
from typing_extensions import Literal,NotRequired
from langgraph.graph.message import add_messages
import json
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint



from langchain_core.prompts import ChatPromptTemplate
import re
if not os.environ['AZURE_OPENAI_API_KEY']:
    os.environ['AZURE_OENAI_API_KEY']=os.environ['AZURE_OPENAI_API_KEY']


if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.environ["HUGGINGFACEHUB_API_TOKEN"]

logging.basicConfig(level=logging.DEBUG)
llm=AzureChatOpenAI(
    azure_endpoint=os.environ['AZURE_ENDPOINT'],
    azure_deployment=os.environ['AZURE_DEPLOYMENT'],
    openai_api_version=os.environ['AZURE_API_VERSION'],
    temperature=0.1
)

# model = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     temperature=0.1,
#     max_new_tokens=1024,
# )
# llm = ChatHuggingFace(llm=model)


#create a state
class State(TypedDict):
    messages:Annotated[list[HumanMessage|AIMessage],add_messages]
    df_dict:dict
    file_path:str
    user_question:NotRequired[str]
    schema:NotRequired[dict]
    code:NotRequired[str]
    text_response:NotRequired[str]
    visuals:NotRequired[list]
    explained_answer:NotRequired[str]
    classification:NotRequired[str]
    error:NotRequired[str]
    final_answer :NotRequired[str]

# define  class for structured_output
class classifyQuery(BaseModel):
    classified:Literal["not_related_to_schema", "related_to_schema"]=Field(
        ... #indicates required , if none is used then it means its optional
        ,description="Decide if the question asked is related to the schema or not "
    )

# class answer_and_followup(BaseModel):
#     answer:str=Field(...,description="This is the answer to the question. The final_answer ")
#     follow_up_prompt:str=Field(...,description="a follow up prompt/question the user would ask")


evaluator_llm=llm.with_structured_output(classifyQuery)   #llm with structured output
# answer_llm=llm.with_structured_output(answer_and_followup)
#prompts 

#evaluate if the question is related to the df or not
intent_prompt=ChatPromptTemplate.from_messages([
    ("system","""You are an intelligent python assistant that understands the schema of a specific pandas DataFrame ('df') .Given a DataFrame schema and a user's question, determine if the question can be answered using the DataFrame or  not
             Only classify as 'related_to_schema' if the schema clearly includes the data required to answer the question otherwise classify as 'not_related_to_schema',
           "a question can use  attributes, characters,/ characteristics to mean column of a dataset , so be careful in classifying such a question "
           """),
    ("human","schema:{schema}, user_question:{user_question}")]
)

# query llm for code
system_prompt=ChatPromptTemplate.from_messages([
        ("system","""
You are an intelligent Data analysis Python assistant that understands the schema of a specific pandas DataFrame (`df`) and generates accurate, executable Python code to answer user questions.

The dataset is loaded in a DataFrame named `df`. 


======================
⚙️ CODE GENERATION RULES
======================
1. Generate ONLY clean, executable Python code
2. Use the DataFrame `df` — it's already loaded
3. Always print or display the final result
4. Handle missing values and edge cases
5. Add brief comments to explain non-trivial logic
6. For visualizations, use `matplotlib.pyplot` or `seaborn`
   - Always use `plt.figure(figsize=(6,4),dpi=100)` for every visual
   - Always call `plt.savefig(...)` 
   -Alongside every visualization, generate Python code that produces a textual summary of the same insight
   - do not use  plt.title
7. Use appropriate error handling where necessary
8.always start a new line after am executable line like this print()\\n , DO NOT use embedded newline characters like print('\\n...') to force line breaks."



====================
✅ RESPONSE FORMAT
====================
- Your response must be a single valid Python code block
- Do NOT include any explanation, prose, or markdown outside the code
- Wrap the code in triple backticks like this: ```python ... ```

Now, generate the correct Python code to answer the question.

Here are inputs for the 'df' and the 'user_question'
"""   
         ),
         ("human","dataframe schema:{schema},\n user_question:{user_question} ")
         ])

#refine answer prompt
def get_explained_prompt(question, code, text_response,visualization_response):
    text_content=f"""
    You are a smart data analyst.

Your job is to explain **in simple, human language** what the result of the analysis means.
You will be having the initial question, the python code used for the analysis , the textual output for the analysis and the visualizations generated if any is generated 

 INSTRUCTIONS
- write a clear and concise explanation of what the result shows.
- Explain patterns or insights from text and visualizations.
- Use normal human words, not technical jargon.
- You can reference variable names like 'price', 'rating', etc.
- If there's a correlation, describe it.

====================
 USER QUESTION
====================
    {question}

====================
 PYTHON CODE USED
====================
{code}

====================
 TEXTUAL OUTPUT
====================
{text_response}

====================
VISUALIZATIONS
====================

"""
    content=[
        {"type":"text",
         "text":text_content
         }
    ]
    if visualization_response:
        for i,visual in enumerate(visualization_response):
            image_url=f"data:image/png;base64,{visual}"
            image_content={
                "type":"image_url",
                "image_url":{"url":image_url}
            }
            content.append(image_content)
    return HumanMessage(content=content)


#online_prompt
online_prompt_template=ChatPromptTemplate.from_messages([
    ("system","You are intelligent assistant that answers correctly and intelligently the users question.Please provide an accurate answer given the users question"),
    ("human","user_question:{user_question}")]
)


# function to clean the code

#clean the code
def clean_code(python_code:str)->str:
    #extract code from he backticks
    code_match=re.search(r"```(?:python)?\n(.*?)```",python_code,re.DOTALL)
    if code_match:
        raw_code=code_match.group(1)
    else:
        raw_code=python_code
    #clean the code

    cleaned_code=raw_code.encode('utf-8').decode('unicode_escape')

    return cleaned_code.strip()

def safe_dataframe_conversion(df_dict):
    """Convert df_dict to DataFrame with serialization-safe types"""
    df = pd.DataFrame(df_dict)
    # Use convert_dtypes for consistent, safe type conversion
    df_converted = df.convert_dtypes()
    # df = df.applymap(lambda x: x.item() if hasattr(x, 'item') else x)
    return df_converted

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    # elif isinstance(obj, np.integer):
    #     return int(obj)
    # elif isinstance(obj, np.floating):
    #     return float(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj

#get the document
def get_df(file_path)->str:
    try:
       df=pd.read_csv(file_path)
       return {"df":df}
    except Exception as e:
        logging.error(f'Error-1 in getting the : {e}',)
        return {"error":f"Error-1 in getting the : {e}"}

#create a schema
def create_schema(df,file_path)->dict:
    try:
        schema={
    "file_name":Path(file_path).name,   
    "shape":df.shape,
    "columns":df.columns.to_list(),
    "sample_data":df.head(3).to_dict('records'),
    # "data_types":df.dtypes.to_dict(),
    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
    "unique_counts":  {col: int(df[col].nunique())   for col in df.columns},

    # "missing_values":df.isna().sum().to_dict(),
    "non_null":df.notnull().sum().to_dict(),
    # "unique_counts":df.nunique().to_dict(),   
    "numeric_columns":df.select_dtypes(include=[np.number]).columns.to_list(),
    "categorical_columns":df.select_dtypes(include=['object']).columns.to_list(),
    "date_columns":df.select_dtypes(include=['datetime']).columns.to_list(),
    "column_description":df.describe(include="all").to_dict(),
    "categorical_samples": {
        col: df[col].dropna().value_counts().head(3).to_dict()
        for col in df.select_dtypes(include='object').columns
    },
    "memory_usage_bytes":df.memory_usage(deep=True).sum()
            }
        
        return schema
    
    except Exception as e:
        logging.error(f"Error-2 error in creating the schema: {e}")
        return {"error":f"Error-2 error in creating the schema: {e}"}



def generate_intent(state:State):
    try:
        if 'df_dict' not in state or state['df_dict'] is None or not state['df_dict']:
            logging.warning(f"WARNING there is no dataframe in the agent")
            return {"error":f"No dataframe in the agent , please upload a dataframe first"}
            
        df=safe_dataframe_conversion(state['df_dict'])
        file_path=state.get("file_path")

        if 'schema' not in state or state['schema'] is None or not state['schema']:
            df_schema=create_schema(df=df,file_path=file_path)
        
        else:
            df_schema=state['schema']
        # df_schema=create_schema('df')
        # schema=json.dumps(df_schema,indent=2)
        user_question=state['messages'][-1].content
        evaluator_prompt=intent_prompt.invoke({"schema":df_schema,"user_question":user_question})
        response=evaluator_llm.invoke(evaluator_prompt)
        print(f"CLASSIFICATIONM RESPONSE : {response}\n\n" ,response.classified)
        return convert_numpy_types({"classification":response.classified,"user_question":user_question,"schema":df_schema})
    except Exception as e:
        logging.error(f"Error-3 error in getting the intent: {e}")
        return {"error":f"Error-3 error in getting the intent: {e}"}
        


def route_decision(state:State):
    try:
        if state['classification'].lower()=='related_to_schema':
            return 'continue_to_analyze'
        elif state['classification'].lower()=='not_related_to_schema':
            return 'search_online'
        else:
            logging.warning(f"unexpected classification : {state.get("classification","")}")
            return "unknown_path"

    except Exception as e:
        logging.error(f"Error-4 error in routing  the question: {e}")
        return {"error":f"Error-4 error in routing the question: {e}"}
    


def get_response_from_csv(state:State)->State:
    try:
        if 'df_dict' not in state or state['df_dict'] is None or not state['df_dict']:
            logging.warning(f"WARNING there is no dataframe in the agent")
            return {"error":f"No dataframe in the agent , please upload a dataframe first"}
            
        df=safe_dataframe_conversion(state['df_dict'])
        #pass the schema and querry llm to generate the python code
        code_prompt=system_prompt.invoke({"schema":state['schema'],"user_question":state['user_question']})
        llm_code=llm.invoke(code_prompt)

        print(f"LLM_CODE : \n\n {llm_code.content}")
        #clean the code
        executable_code=clean_code(python_code=llm_code.content)

        if not hasattr(plt,'_original_savefig'):
            plt._original_savefig=plt.savefig

        exec_globals={'df':df,'plt':plt,'sns':sns}
        image_b64_lists=[]

         #this function is patches the savefig and encodes the imaage/ extract the visauals
        def patch_savefig(*args,**kwargs):
            # print("entered)"
            capture_kwargs=kwargs.copy()
            if 'format' not in capture_kwargs:
                capture_kwargs['format']='png'
            buf=BytesIO()
            plt._original_savefig(buf,**capture_kwargs)
            buf.seek(0)
            encod_image=base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            image_b64_lists.append(encod_image)
            # print(f"Image captured! Total images: {len(image_b64_lists)}")
            return encod_image

        plt.savefig=patch_savefig
          #get the string text
        old_stdout=sys.stdout
        sys.stdout=mystdout=StringIO()
        try:
            # execute the code
            exec(executable_code,exec_globals)
            output_text=mystdout.getvalue()
        finally:
            sys.stdout=old_stdout
            plt.savefig=plt._original_savefig
        
        print(f"OUTPUT_TEXT \n\n {output_text} \n\n VISUALIZATION \n\n {image_b64_lists}")

            #explain the prompts
        explained_prompt=get_explained_prompt(question=state['user_question'],code=executable_code,text_response=output_text,visualization_response=image_b64_lists)
        explained_answer=llm.invoke([explained_prompt])
        final_answer=explained_answer.content

        
        return {"text_response":output_text,"visuals":image_b64_lists,"final_answer":final_answer}
    
    except Exception as e:
        # logging.error(f'Error-1 in getting the response from the csv : {e}',)
        return {"error":f"Error-1 in getting the response from the csv : {e}"}


def get_response_from_online(state:State):
    try:
        online_prompt=online_prompt_template.invoke({"user_question":state['user_question']})
        explained_answer=llm.invoke(online_prompt)
        print(f"ANSWER : {explained_answer}")
        final_answer=explained_answer.content 
        print(f"final_answer :: {final_answer}")
        return {"final_answer":final_answer}
    except Exception as e:
        # logging.error(f'Error-6 in getting the response from online : {e}',)
        return {"error":f"Error-6 in getting the response from online : {e}"}
    


#build the agent
try:
    #add nodes
    agent_builder=StateGraph(State)
    agent_builder.add_node('generate_intent',generate_intent)
    agent_builder.add_node('get_response_from_csv',get_response_from_csv)
    agent_builder.add_node('get_response_from_online',get_response_from_online)

    #add edges
    agent_builder.add_edge(START,'generate_intent')
    agent_builder.add_conditional_edges(
        'generate_intent',
        route_decision,
        {
            "continue_to_analyze":'get_response_from_csv',
            "search_online":'get_response_from_online'
        }
        
    )

    agent_builder.add_edge('get_response_from_csv',END)
    agent_builder.add_edge('get_response_from_online',END)
    
    # checkpointer=MongoDBSaver.from_conn_string(DB_URL)
    # with MongoDBSaver.from_conn_string(DB_URL) as checkpointer:
    # checkpointer.setup()
    analysis_agent = agent_builder.compile(checkpointer=MemorySaver())
    def query_analysis_agent(file_path,question,thread_id):
        try:
            config = {
                "configurable": {
                "thread_id": thread_id
            }
            }
            print(f"file_path..{file_path}")
            df = pd.read_csv(rf"{file_path}")
        # df = df.astype(object)
            df=df.convert_dtypes()
            df_dict = df.to_dict('records')
            df_dict = [convert_numpy_types(record) for record in df_dict]
            agent_response=analysis_agent.invoke({
                "messages":[HumanMessage(content=question)],
                "df_dict":df_dict,
                "file_path":file_path
            },config)
            if agent_response.get("error"):
                return {"error":f"ERROR in getting response:: \n\n {agent_response.get("error")}"}

            return {
                "text_response":agent_response.get("text_response",""),
                "visuals":agent_response.get("visuals",[]),
                "final_answer":agent_response.get("final_answer","")
            }

                # checkpointer.close()
        except Exception as e:
            logging.error(f" ERROR  IN QUERRYING the agent::: {e}")



except Exception as e:
    logging.error(f'Error-8 in building the agent: {e}')
    raise

if __name__ == "__main__":
    config = {
        "configurable": {
        "thread_id": "1233"
               }
            }
    file_path="/home/spencer/Downloads/housepricedata.csv"
    thread_id="1233"

    while True:
        input_question = input("question: ")
        if input_question.lower() == 'exit':
            break

    # df = pd.read_csv(r"/home/spencer/Downloads/housepricedata.csv")
        # df = df.astype(object)
    # df=df.convert_dtypes()
    # df_dict = df.to_dict('records')
    # df_dict = [convert_numpy_types(record) for record in df_dict]
        try:
        # res = analysis_agent.invoke({
        #     "messages": [HumanMessage(content=input_question)],
        #     "df_dict": df_dict
        #     }, config)
            res=query_analysis_agent(question=input_question,thread_id=thread_id,file_path=file_path)

            if not res.get('error'):
                print(f"\nTEXT_RESPONSE:\n{res.get('text_response') or 'No text response recorded'}")
                print(f"\nVISUALIZATIONS:\n{res.get('visuals') or 'No visualizations generated'}")
                print(f"\nEXPLAINED_ANSWER:\n{res['final_answer']}")
            else:
                logging.error(f"AGENT_ERROR: \n{res['error']}")

        except Exception as inner_e:
            logging.error(f"Invocation error: {inner_e}")
            continue
        # finally:
        #     try:
        #         checkpointer.client.close()
        #     except Exception as e:
        #         logging.error(f"error in closing up")
            