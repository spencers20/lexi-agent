from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import getpass
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START,StateGraph
from langgraph.checkpoint.memory import InMemorySaver
# from langchain import hub
from IPython.display import display,Image
from langchain_openai import AzureChatOpenAI
import asyncio
import uuid
# from pathlib import


load_dotenv()
LANGSMITH_TRACING=True
# LANGSMITH_ENDOINT=os.environ['LANGSMITH_ENDPOINT']

# LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']

# LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']





if not os.getenv("COHERE_API_KEY"):
    os.environ['COHERE_API_KEY']=os.environ['COHERE_API_KEY']

if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ['AZURE_OPENAI_API_KEY']=os.environ['AZURE_API_KEY']



# vectore store fron pinecone 
pc=Pinecone(api_key=os.environ['PINECONE_API'])
index=pc.Index('lexifile')
embeddings=CohereEmbeddings(model="embed-english-v3.0")



memory=InMemorySaver()

# llm=ChatGroq(
#     api_key=os.environ['GROQ_API_KEY'],
#     model='llama-3.3-70b-versatile',
#     temperature=0.7
# )

llm=AzureChatOpenAI(
    
    azure_endpoint=os.environ['AZURE_ENDPOINT'],
    azure_deployment=os.environ['AZURE_DEPLOYMENT'],
    openai_api_version=os.environ['AZURE_API_VERSION'],
    temperature=0.6
)

# retriever_template="""
#            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
#            If the answer is not explicitly found, but can be inferred, respond intelligently and say it is not directly stated.  
#             If you cannot answer or infer from the context, politely reply with: "Not provided in the context."
#              Use three sentences maximum and keep the answer concise.
             
#             Only if the answer was explicitly found or inferred from the context, politely add a follow-up question or suggestion that is relevant to the user's question and based on the information found in the context. The follow-up should sound natural and helpful, not like a system label.
#             Question: {question} 
#             Context: {context} 
#             Answer:
#          """

# Intent template
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intent detection system. Your job is to extract the user's intent from their response, taking into account any follow-up prompt if provided, and formulate another question (intent question).

Instructions:
- If a follow-up prompt is provided **AND RELATED to users response**, use it along with the user's response to determine the intent.
- If **no follow-up prompt is given OR NOT RELATED to users response**, determine the intent from the user's response alone.
- Intent should  not be a question but a clear understandable statement  
- Return ONLY and ONLY the intent.
- DO NOT provide any explanation, reasoning, or extra text. Just return the intent."""),
    ("human", """
     follow-up prompt: {followupprompt}
     user response: {question}
     Intent:"""
    )
])

retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("human", "Question: {question}\n\nContext: {context}")
])

# Follow-up prompt template
follow_up_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a follow-up prompt generator system. You are tasked with generating a follow-up prompt given the user's intent and the context.

Instructions:
- The follow-up prompt should be clearly related to both the intent and the context. Do not go outside the scope.
- Use words or phrases found in the context to ground the prompt.
- Keep the follow-up prompt short (one sentence), clear, concise, and easy to understand.
- Be creative. You can use formats like: 
  - "Would you like me to ..."
  - "Should I explain more on..."
  - "Let me know if I can help you with..."
- Return ONLY AND ONLY the follow-up prompt.
- DO NOT provide any explanation, reasoning, or extra text. JUST THE FOLLOW-UP PROMPT"""),
    ("human", """
     intent: {intent}  
     context: {context}

Follow-up Prompt:""")
])

# retriever_prompt=hub.pull("rlm/rag-prompt")

# class to define the states used in the agent
class State(TypedDict):
    question:str
    context:list[Document]
    answer:str
    intent:str
    followupprompt:str
    final_answer:str
    namespace:str

# node to geneerate the intent
async def intent_generator(state:State):
    try:
        final_intent_prompt=await intent_prompt.ainvoke({"followupprompt":state.get("followupprompt",""),"question":state["question"]})
        intent=await llm.ainvoke(final_intent_prompt)
        print("intent",intent.content)
        return {"intent":intent.content}
    except Exception as e:
        return {"final_answer":f"error in getting the intent: {e}"}
    
# node to retrieve from the llm
async def retriever_generator(state:State):
    try:
        if not state["namespace"]:
            return {"error":"enter namespace"}
        vectore_store=PineconeVectorStore(embedding=embeddings,index=index,namespace=state["namespace"])
        retrieved_docs=vectore_store.similarity_search(state["intent"])
        final_retrieve_docs="\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt=await retriever_prompt.ainvoke({"question":state["intent"],"context":final_retrieve_docs})
        response=await llm.ainvoke(final_prompt)
        return {"answer":response.content,"context":final_retrieve_docs}
    except Exception as e:
        return {"final_answer":f"error in retrieving from db {e}"}

#node to generate the follow up prompt
async def follow_up_prompt_generator(state:State):
    try:

        final_prompt=await follow_up_prompt.ainvoke({"intent":state["intent"],"context":state["context"]})
        followup=await llm.ainvoke(final_prompt)
        final_answer=state["answer"].strip() + "\n\n"+ followup.content.strip()
        return {"final_answer":final_answer, "followupprompt":followup.content}
    except Exception as e:
        return {"final_answer":f"error in generating follow up prompt {e}"}

try:
    agent_builder= StateGraph(State).add_sequence([intent_generator,retriever_generator,follow_up_prompt_generator])
    agent_builder.add_edge(START,'intent_generator')
    retriever_agent=agent_builder.compile(checkpointer=memory)
except Exception as e:
     {"final_answer ":f"error in building the agent {e}"}


async def query_agent(question,namespace,thread_id):
    try:
        display(Image(retriever_agent.get_graph().draw_mermaid_png()))
        config={"configurable":{"thread_id":thread_id}}
        answer=await retriever_agent.ainvoke({"question":question, "namespace":namespace},config)
        return answer["final_answer"]
    except Exception as e:
        return {"final_answer":f"error in querying the agent {e}"}


# async def main():
    
#     while True:
#         question=input("question: ")
#         if question.lower()=='exit':
#             break
#         res=await query_agent(question=question, namespace=namespace,thread_id=thread_id)
#         print(f"answer : {res}")

# retriever_agent=agent_builder.compile()
    
# if __name__=="__main__":
#     asyncio.run(main())
   
    