from langchain_pinecone import PineConeVectorStore
from langchain_openai import AzureChatOpenAI
from Typing.extensions import TypedDict,Annotated
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain.core_messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import logging
from pinecone import Pinecone
from cohere import CohereEmbeddings
from langchain.core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START


if not os.getenv("COHERE_EMBEDDINGS"):
    os.environ['COHERE EMBEDDINGS']=os.environ['COHERE EMBEDDINGS']

if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"]=os.environ["AZURE_OPENAI_API_KEY"]

if not os.getenv("PINECONE_API"):
    os.environ['PINECONE_API']=os.environ['PINECONE_API']

pc=Pinecone(api_key=os.environ['PINECONE_API'])
index=pc.Index('Afyasphere')
embeddings=CohereEmbeddings(model="embed-english-v3.0")


logging.basicConfig(level=logging.DEBUG)
load_dotenv()


vectorstore=PineConeVectorStore(embeddings=embeddings,index=index)
class State(TypedDict):
    messages:Annotated[list[HumanMessage | AIMessage],add_messages]
    context:list[Document]
    answer:list[str]

class dayactivity(BaseModel):
    day:int =Field(...,description='Index of this step in the journey."')
    focus:str =Field(...,description='2-3 word Short title summarizing the health goal for the day ')
    description:str=Field(...,description='A 2-4 line paragraph explaining the intent behind the focus for the day')
    activities:list[str]=Field(...,description='A list of practical, specific actions for the day to support the focus')

class schedule(BaseModel):
    daily_activities:list[dayactivity]=Field(...,description='Ordered list of daily activity objects')

llm=AzureChatOpenAI(
    
    azure_endpoint=os.environ['AZURE_ENDPOINT'],
    azure_deployment=os.environ['AZURE_DEPLOYMENT'],
    openai_api_version=os.environ['AZURE_API_VERSION'],
    temperature=0.6
)
journey_llm=llm.with_structured_output(schedule)


activities_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a health journey activity generator. "
        "Your job is to use the provided `guideline_context`, `journey_title`, `journey_description`, and `number_of_days` "
        "to generate a personalized health journey plan. "
        "For each day (starting from Day 1), include:\n"
        "- Day label (e.g., 'Day 1')\n"
        "- Focus (a short phrase, e.g., 'Cardio Kick', 'Strength Focus')\n"
        "- Description (brief explanation of that day's focus and its benefits)\n"
        "- Associated Activities of Focus (a list of practical daily activities, such as workout, meals, movement, mindfulness, etc., aligned with the focus)\n"
        "Use the `guideline_context` to ensure the generated plan follows scientifically valid health and wellness principles (e.g., exercise variety, recovery balance, nutritional considerations, behavior change).\n"""
     
    ),
    (
        "human",
        """guideline_context:\n{context}\n\n"
        "journey_title:\n{journey_title}\n\n"
        "journey_description:\n{journey_description}\n\n"
        "number_of_days:\n{number_of_days}"""
    )
])

def generatehabits(state:State):
    try:
        vectorstore=PineConeVectorStore(embeddings=embeddings,index=index,namespace=os.environ['HABITS_NAMESPACE'])
        habits_docs=[]
        habits_query=[
            "what are the quidelines to create habits",
            "which habits examples are there"
        ]
        for query in habits_query:
            docs=vectorstore.similarity_search(query,k=10)
            habit_doc='\n\n'.join([doc.page_content for doc in docs])

            habits_docs.extend(hd for hd in habit_doc if hd not in habits_docs)
        generated_habits=habits_prompt.invoke({"guideline_docs":habits_docs})
        habits=llm.invoke(generated_habits).content

        return {"answer":habits}


    
    except Exception as e:
        logging.error(f"ERROR IN GENERATING HABITS:\n {e}")

def generateactivities(state:State):
    try:
        all_docs=[]
        all_docs_query=[
            "health journey creation guidelines and core design principles",
            "daily template components for generated health journey activity ",
            "Example 7 –day health journey  Scheduled activities with Focus",
            " Structure for daily Journey Components Summary"
        ]

        for query in all_docs_query:
            retrieved_doc=vectorstore.similarity_search(query, k=50)
            retrieve_docs='\n\n'.join([doc.page_content for doc in retrieved_doc])
            all_docs.extend(doc for doc in retrieve_docs if doc not in all_docs)

        get_activities_prompt=activities_prompt.invoke({"context":all_docs,"journey_title":journey_title,"journey_description":journey_description,"number_of_days":number_of_days})
        generated_activities=journey_llm.invoke(get_activities_prompt).content.daily_activities
        return {"answer":generated_activities}

        
    except Exception as e:
        logging.error(f'ERROR IN GENERATION :\n{e}')
try:
    agent_builder=StateGraph(State)
    agent_builder.add_node("generatehabits",generatehabits)
    agent_builder.add_node("generateactivites",generateactivities)
    # agent_builder.add_edge(START,generateactivities)
    agent_builder.add_conditional_edge
    agent=agent_builder.compile()
except Exception as e:
    logging.error(f"ERRORR IN BUILDING AGENT: \n{e}")    