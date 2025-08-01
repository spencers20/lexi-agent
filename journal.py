from langchain_pinecone import PineConeVectorStore
from langchain_openai import AzureChatOpenAI
from Typing.extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class State(TypedDict):
    messages:Annotated[list[HumanMessage | AIMessage],addmessages],
    context=list[Document]

class dayactivity(BaseModel):
    day:int =Field(...,description='Index of this step in the journey."')
