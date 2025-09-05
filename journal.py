from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI
from typing_extensions import NotRequired
from typing import TypedDict,Annotated
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
# from langchain_core_messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import logging
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END



if not os.getenv("COHERE_API_KEY"):
    os.environ['COHERE_API_KEY']= os.getenv('COHERE EMBEDDINGS')

if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API_KEY") 

# if not os.getenv("PINECONE_API"):
#     os.environ['PINECONE_API']=os.environ['PINECONE_API']

pc=Pinecone(api_key=os.environ['PINECONE_API'])
index=pc.Index('afyasphere-wellbeing')
embeddings=CohereEmbeddings(model="embed-english-v3.0")


logging.basicConfig(level=logging.DEBUG)
load_dotenv()



class State(TypedDict):
    # messages:Annotated[list[HumanMessage | AIMessage],add_messages]
    journey_title:NotRequired[str]
    journey_description:NotRequired[str]
    number_of_days:NotRequired[str]
    habit_query:NotRequired[str]
    journey_context:list[Document]
    journey_answer:NotRequired[list[str]]
    habit_answer:NotRequired[list[str]]
    prescription:NotRequired[str]
    prescription_answer:NotRequired[list[str]]

class dayactivity(BaseModel):
    day:int =Field(...,description='Index of this step in the journey."')
    focus:str =Field(...,description='2-3 word Short title summarizing the health goal for the day ')
    description:str=Field(...,description='A 2-4 sentences explaining the intent behind the focus for the day')
    activities:list[str]=Field(...,description='A list of practical, specific actions for the day to support the focus')

class schedule(BaseModel):
    daily_activities:list[dayactivity]=Field(...,description='Ordered list of daily activity objects')

class habit_structure(BaseModel):
    habit:str=Field(...,description="a 2-4 word noting the health habit one should adopt")
    habit_description:str=Field(...,description='a 2-4 sentences describing the health habit and its importance')
class habitschedule(BaseModel):
    habits:list[habit_structure] =Field(...,description="ordered list of habits ")

class prescription_schedule(BaseModel):
    dosage:str=Field(...,description=" the amount to be taken each time")
    frequency:int=Field(...,description="how many times per day the dosage should be taken")
    # hours:int=Field(...description="after how many hou per day the dosage should be taken")
    no_of_days:int=Field(...,description="the total number of days the dosage will be taken")



llm=AzureChatOpenAI(
    
    azure_endpoint=os.environ['AZURE_ENDPOINT'],
    azure_deployment=os.environ['AZURE_DEPLOYMENT'],
    openai_api_version=os.environ['AZURE_API_VERSION'],
    temperature=0.6
)
journey_llm=llm.with_structured_output(schedule)
habit_llm=llm.with_structured_output(habitschedule)
prescription_llm=llm.with_structured_output(prescription_schedule)



activities_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a health journey activity generator. "
        "Your job is to use the provided `guideline_context`, `journey_title`, `journey_description`, and `number_of_days` "
        "to generate a personalized health journey plan.  "
        "number_of_days represent how many personalized plans  should be there , provide a personalized plan , if 30 then give for 30 days , if 10 give for 10 days
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

prescription_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a medical assistant.
        Your task is to extract exactly  three values from the given medication prescription statement.

            dosage: <string or null>,
            frequency: <integer or null>,
            no_of_days: <integer or null>
        

        ### Field definitions:
        - **dosage**: The quantity taken each time. Include both the number and the unit if available (e.g., "1 tablet", "30 ml", "2 capsules").  
        - **frequency**: How many times per day the dosage should be taken. Must be a number. If given as hours (e.g., "every 8 hours"), convert to daily frequency using `24 ÷ hours`.  
        - **no_of_days**: The total number of days the dosage will be taken. Must be a number.

        ### Rules:
        1. **Shorthand numeric patterns**:
            - If format is `A*B` or `A x B`, interpret:
                - A = dosage (numeric value, unit if provided elsewhere in the statement)
                - B = frequency (times per day) / (turn any hours to times per day by dividing(24/number_of_hours))
                - no_of_days = null
            - If format is `A*B*C` or `A x B x C`:
                - A = dosage
                - B = frequency
                - C = no_of_days
            1 Teaspoon = 5ml
        2. **Text patterns**:
            -Example: "`2*3 or `2 x 3 ` "  → dosage = "2 doses", frequency =3
            -Example: "2 teaspoons four times a day"  → dosage = "2 teaspoons (10ml)  ", frequency =4 (multiply number of teaspoons by 5 to get the mls)
            -Example: "`2capsules *3 or `2capsules x 3 ` "  → dosage = "2capsules", frequency =3
            - Example: "2 tablets twice a day" → dosage = "2 tablets", frequency = 2
            - Example: "30ml after every 8 hours" → dosage = "30 ml", frequency = 24 ÷ 8 = 3
        3. **Written numbers**:
            - Convert written numbers to digits: "five days" → 5
        4. **Missing values**:
            - If a field cannot be determined, set it to `null`.

        Your answer must only contain the JSON object. No explanations.
        """
    ),
    (
        "human",
        "prescription: {prescription}"
    )
])


def route_habits_or_journey(state:State):
    try:
        logging.debug('ROUTING AGENT...')
        habit_query=state.get("habit_query","") 
        prescription=state.get("prescription","")
        if habit_query and habit_query.strip():
            logging.debug("GETTING TO habits...")
            return "suggest_habit"
        elif prescription and prescription.strip():
            logging.debug("GETTING TO prescription..")
            return "get_prescription"
        else:
            logging.debug('getting to JOURNEYS')
            return "generate_journey"


    except Exception as e:
        logging.error(f'ERROR IN THE CONDITIONAL NODE {e}')

def generatehabits(state:State):
    try:
        vectorstore=PineconeVectorStore(embeddings=embeddings,index=index,namespace=os.environ['HABITS_NAMESPACE'])
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
        habits=habit_llm.invoke(generated_habits).habits

        return {"habit_answer":habits}


    
    except Exception as e:
        logging.error(f"ERROR IN GENERATING HABITS:\n {e}")



def generateactivities(state:State):
    try:
        logging.debug('GENERATING ACTIVITIES...')
        vectorstore=PineconeVectorStore(embedding=embeddings,index=index,namespace=  os.environ['JOURNEY_NAMESPACE'])
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

        get_activities_prompt=activities_prompt.invoke({"context" :all_docs,"journey_title":state.get("journey_title",""),"journey_description":state.get("journey_description",""),"number_of_days":state.get("number_of_days","")})
        generated_activities=journey_llm.invoke(get_activities_prompt).daily_activities
        logging.debug(f"GENERATED ACTIVITIES::\n {generated_activities}")
        return {"journey_answer":generated_activities}

        
    except Exception as e:
        logging.error(f'ERROR IN GENERATION :\n{e}')

def generate_prescription(state:State):
    try:
        prescription=state.get("prescription","")
        prescription_prompt=prescription_template.invoke({"prescription":prescription})
        llm_res=prescription_llm.invoke(prescription_prompt)
        logging.debug(f"GOT PRESCRIPTIONS..\n {llm_res}")
        return {"prescription_answer":llm_res}
        
    except Exception as e:
        logging.error(f"GENERATING PRESCRIPTION ERROR /n {e}")
try:
    agent_builder=StateGraph(State)
    agent_builder.add_node("generatehabits",generatehabits)
    agent_builder.add_node("generateactivities",generateactivities)
    agent_builder.add_node("generate_prescription",generate_prescription)
    # agent_builder.add_edge(START,generateactivities)
    agent_builder.add_conditional_edges(
        START,
        route_habits_or_journey,{
            "suggest_habit":"generatehabits",
            "get_prescription":"generate_prescription",
            "generate_journey":"generateactivities"
        }
    )
    agent_builder.add_edge("generatehabits", END)
    agent_builder.add_edge("generateactivities",END)
    agent_builder.add_edge("generate_prescription",END)
    agent=agent_builder.compile()
    def get_habits_or_journeys(habit_query,journey_title,journey_description,number_of_days,prescription):
        try:
            logging.debug('INVOKING AGENT...')
            print(f'agent...\n {agent}')
            answer=agent.invoke({"habit_query":habit_query,
                                 "journey_description":journey_description,
                                 "journey_title":journey_title,
                                 "number_of_days":number_of_days,
                                 "prescription":prescription
                                 })
            if answer.get("habit_answer") is not None and answer.get("journey_answer") is None:
                return answer.get("habit_answer","")
            elif answer.get("habit_answer") is None and answer.get("journey_answer") is  not None:
                return answer.get("journey_answer","")
            else:
                return answer.get("prescription_answer")

        except Exception as e:
            logging.error(f"ERRORR IN QUERYING AGENT: \n{e}")
except Exception as e:
    logging.error(f"ERRORR IN BUILDING AGENT: \n{e}")    

  

if __name__=="__main__":
    try:
        while True:
            habit_query=input("habit_query: ")
            journey_title=input("Journey_title: ")
            journey_description=input("Journey_description: ")
            number_of_days=input("Number_of_days: ")
            prescription=input("Prescription: ")
            if any(val.strip().lower()=='exit' for val in [habit_query,journey_title,journey_description,number_of_days,prescription]):
                print('Exiting the program...')
                break
            response=get_habits_or_journeys(habit_query=habit_query,journey_title=journey_title,journey_description=journey_description,number_of_days=number_of_days,prescription=prescription)
            if response is None:
                break
            print(f'HABITS/ JOURNEY GENERATED:: \n {response}')
           
    except Exception as e:
        logging.error('ERROR IN MAIN FUNCTION')