from langchain.pydantic_v1 import BaseModel, Field


class ResponseObject(BaseModel):
    """
    Class which defines the output format of the llm request
    """

    reply1: str = Field(description="The first generated reply")
    reply2: str = Field(description="The second generated reply")
    reply3: str = Field(description="The third generated reply")
