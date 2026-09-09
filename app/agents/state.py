from typing import TypedDict,List,Annotated
import operator


class AgentState(TypedDict):
    # Using Annotated with operator .add ensures that messages are added to the history instead of replacing it
    messages:Annotated[List[dict], operator.add]
    current_query:str
    documents:List[str]
    plan:List[str]
    status:str
    final_answer:str