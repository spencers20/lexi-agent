from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
# import getpass
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
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
# from langchain_community.chat_models.huggingface import ChatHuggingFace
# from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# from pathlib import


load_dotenv()
LANGSMITH_TRACING=True
# LANGSMITH_ENDOINT=os.environ['LANGSMITH_ENDPOINT']

# LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']

# LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']





# if not os.getenv("COHERE_API_KEY"):
#     os.environ['COHERE_API_KEY']=os.environ['COHERE_API_KEY']

# if not os.getenv("AZURE_OPENAI_API_KEY"):
#     os.environ['AZURE_OPENAI_API_KEY']=os.environ['AZURE_API_KEY']

# if not os.getenv("OPENAI_API_KEY"):
#     os.environ['OPENAI_API_KEY']=os.environ['OPENAI_API_KEY']

# if not os.getenv("ANTHROPIC_API_KEY"):
#     os.environ["ANTHROPIC_API_KEY"]=os.environ["ANTHROPIC_API_KEY"]

# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.environ["HUGGINGFACEHUB_API_TOKEN"]

# vectore store fron pinecone 
pc=Pinecone(api_key=os.environ['PINECONE_API'])
index=pc.Index(os.environ['LEXIFILE_INDEX'])
embeddings=CohereEmbeddings(model="embed-english-v3.0")



memory=InMemorySaver()

# llm=ChatGroq(
#     api_key=os.environ['GROQ_API_KEY'],
#     model='llama-3.3-70b-versatile',
#     temperature=0.7
# )

# llm=AzureChatOpenAI(
    
#     azure_endpoint=os.environ['AZURE_ENDPOINT'],
#     azure_deployment=os.environ['AZURE_DEPLOYMENT'],
#     openai_api_version=os.environ['AZURE_API_VERSION'],
#     temperature=0.6
# )
# llm=ChatOpenAI(
#     model="gpt-4.1-nano",
#     temperature=0.6
# )
# llm= init_chat_model("claude-sonnet-4-5-20250929")

model = HuggingFaceEndpoint(
    repo_id=os.environ["MODEL_ID"],
    temperature=0.6,
    max_new_tokens=1024,
)
llm = ChatHuggingFace(llm=model)

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

# lesson plan prompt template
# lessonplan_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# You are an expert CBC (Kenya) teacher and curriculum specialist trained in KICD standards.

# Your task is to generate a CBC-compliant lesson plan for a SINGLE 40-minute lesson.

# Context usage rules (VERY IMPORTANT):
# - The KICD curriculum design context is used ONLY to guide:
#   • the lesson plan structure
#   • required components
#   • CBC compliance and standards
# - The subject context represents the learner’s textbook and provides:
#   • content knowledge
#   • concepts, examples, and facts related to the lesson topic
# - Do NOT invent content outside the subject context.

# You will be given:
# a) KICD curriculum design context (structure and standards)
# b) Subject context (textbook content)
# c) Lesson topic (the specific sub-sub strand to be taught within the 40 minutes)

# Follow CBC principles:
# - Learner-centered teaching
# - Competency-based learning
# - Age-appropriate language
# - ONE main key competency per lesson

# Generate the lesson plan using the EXACT structure below:

# 1. Specific Learning Outcomes (end of the 40 minutes)
#    - Write 2–3 clear, observable outcomes using action verbs
#    - Outcomes must be achievable within one lesson

# 2. Key Competency
#    - Select ONE primary CBC key competency
#    - Explain briefly how learners will demonstrate it during the lesson

# 3. Learning Experiences
#    a) Introduction (first 3–5 minutes)
#       - Describe what the teacher does to activate prior knowledge and introduce the lesson

#    b) Learner Activities
#       - Describe step-by-step what learners do
#       - Activities must be learner-centered and support the selected key competency

#    c) Teacher Activities
#       - Describe how the teacher facilitates, guides, and supports learning

# 4. Learning Resources
#    - List appropriate resources based on the subject context

# 5. Assessment
#    - Describe how the teacher assesses learning during the lesson
#    - Assessment must align with the learning outcomes and key competency

# 6. Conclusion
#    - Reflection: (leave this section blank for the teacher to complete after the lesson)
#    - Other activities: recap key points and assign relevant homework
# """
#     ),
#     (
#         "human",
#         """
# KICD Curriculum Design Context (Guides structure and standards):
# {KICD_Context}

# Subject Context (Textbook content for the lesson topic):
# {subject_context}

# Lesson Topic (Sub-sub strand to be taught in this 40-minute lesson):
# {lesson_topic}
# """
#     )
# ])

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
    # strand:str    #ADDED FOR THE LESSON PLANNER
    # sub_strand:str
    # lesson_topic:str
    # namespaceII:str

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
#ADDED
        # KICD retrieved docs
#         retrieval_query = f"""
# KICD curriculum content for a lesson plan.

# Strand: {state['strand']}
# Sub-strand: {state['sub_strand']}
# Lesson topic (sub-sub strand): {state['lesson_topic']}

# Retrieve ONLY curriculum sections that include:
# - Specific Learning Outcomes
# - Suggested Learning Experiences
# - Core / Key Competencies
# - Suggested Assessment Methods

# Exclude:
# - Curriculum rationale
# - Vision and mission statements
# - National goals of education
# - Other strands or grades
# """

        # KICD_vec=PineconeVectorStore(embedding=embeddings,index=index,namespace=state['namespaceII'])
        # retrieved_docsII=KICD_vec.similarity_search(retrieval_query)
        # KICD_retrieval_docs="\n\n".join([doc.page_content for doc in retrieved_docsII])

        final_prompt=await retriever_prompt.ainvoke({"question":state["intent"],"context":final_retrieve_docs})
        # final_prompt=await lessonplan_prompt.ainvoke({"KICD_Context":KICD_retrieval_docs,"subject_context":final_retrieve_docs,"lesson_topic":state['lesson_topic']})     #ADDED
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
        return {"final_answer":f"error in generating follow up prompt {e} {state["intent"]}"}

# building the agent (bring the agent all together )
try:
    agent_builder= StateGraph(State).add_sequence([intent_generator,retriever_generator,follow_up_prompt_generator])
    agent_builder.add_edge(START,'intent_generator')
    retriever_agent=agent_builder.compile(checkpointer=memory)
except Exception as e:
     {"final_answer ":f"error in building the agent {e}"}

# querying the agent
async def query_agent(question,namespace,thread_id):
    try:
        display(Image(retriever_agent.get_graph().draw_mermaid_png()))
        config={"configurable":{"thread_id":thread_id}}
        answer=await retriever_agent.ainvoke({"question":question, "namespace":namespace},config)
        # answer=await retriever_agent.ainvoke({"question":question, "namespace":namespace,"namespaceII":namespaceII,"lesson_topic":lesson_topic,"sub_strand":sub_strand,"strand":strand},config)
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
   
    