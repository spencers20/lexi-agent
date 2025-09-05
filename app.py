from fastapi import FastAPI,Request
from agent import query_agent
from pydantic import BaseModel
import uvicorn
import uuid
from typing import Optional
from data_analysis_agent import query_analysis_agent
from journal import get_habits_or_journeys
app=FastAPI()

class QueryInput(BaseModel):
    question:Optional[str]=None
    namespace:Optional[str]=None
    file_path:Optional[str]=None
    thread_id:Optional[str]=None
    habit_query:Optional[str]=None
    journey_title:Optional[str]=None
    journey_description:Optional[str]=None
    number_of_days:Optional[str]=None
    prescription:Optional[str]=None

@app.post("/agent")
async def call_agent(query:QueryInput):
    try:
        # if not query.thread_id:
        #     thread_id = str(uuid.uuid4())
        # else:
        #     thread_id=query.thread_id
        # file_path=io.BytesIO(contents)

        thread_id=query.thread_id or str(uuid.uuid4())
        namespace=query.namespace
        file_path=query.file_path
        
        if not namespace  and not file_path:
            return {"error":"No Document loaded"}
            raise
        if namespace:
            answer=await query_agent(question=query.question,namespace=namespace,thread_id=thread_id)
            return {"response":answer, "chatId":thread_id}
        if file_path:
            answer=query_analysis_agent(question=query.question, file_path=file_path,thread_id=thread_id)
            return {"response":answer.get("final_answer"), "text_response":answer.get("text_response"), "visuals":answer.get("visuals"),"chatId":thread_id}


    except Exception as e:
        print("error in querrying agent",e)
        return {"error":f"error querying the agent {e}"}
    
@app.post("/habit_and_journeys")
def call_habit_and_journeys(query:QueryInput):
    try:
        print("calling the journey habit agent...")
        response=get_habits_or_journeys(habit_query=query.habit_query,journey_title=query.journey_title,journey_description=query.journey_description,number_of_days=query.number_of_days,prescription=query.prescription)
        return {"response":response}
    except Exception as e: 
        return{"error":f"error in querrying the habits and journeys agent {e}"}

# @app.post("/analyzer")
# def call_analyzer(query:QueryInput):
#     try:
#         if not query.thread_id:
#             thread_id=str(uuid.uuid4())
#         else:
#             thread_id=query.thread_id

#     except Exception as e:
#         return {"error":f"error in querrying the analyzeerr \n\n {e}"}
if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000, reload=True)

