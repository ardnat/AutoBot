import json
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma 
from langchain.embeddings import OpenAIEmbeddings
import os
from datetime import datetime
from todoist import get_tasks_due_today
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool
import os
from langchain.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import BaseStringMessagePromptTemplate
import requests
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')

def get_llm():
    # Load the chat model
    chat_model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4-1106-preview",
        temperature=0.3,
    )

    return chat_model
        

class LangChainAgent:
    def __init__(self):
        # Initialize your agent here
        tools = [
            
            Tool.from_function(func=TodoistTool().process, name="todoist", description="Ask a question about items in todoist"),
            Tool.from_function(func=EmailTool().process, name="email", description="Send an email to the user when they request it"),
            Tool.from_function(func=SaveContex().process, name="saveLongtermMemory", description="Saves a message from the user to long term memory for future questions. Input: string to save"),
            Tool.from_function(func=ReadContex, name="getLongtermMemory", description="Get long term memory that has been saved for a user. Output: json strings saved in long term memory"),


        ]
        self.agent_chain = initialize_agent(
            tools,
            get_llm(),
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs = {
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory"), MessagesPlaceholder(variable_name="contextInput")],
            },
            contextInput = "Latest context the user wants to remember: "+ReadContex(""),
            memory = ConversationBufferMemory(memory_key="memory", return_messages=True),
        )

    def run(self, input_data):
        # Process the input data and return the result
        print(input_data)
        return self.agent_chain.run({"input":input_data})

class EmailTool:
    def __init__(self):
        return
    def process(self, input_data):
        requests.post(
            "https://maker.ifttt.com/trigger/sendmyselfemail/json/with/key/z9EUiOhmUdn7reKfojoN7",
            data={"data": input_data},
        )
        return "Email sent"
class SaveContex:
    def __init__(self):
        return
    def process(self, input_data):
        # append to the the context file
        with open("context.txt", "a") as myfile:
            myfile.write(f'{{"date":{datetime.now()},"content":{input_data} \n }}')
        return "Print this entire string out as the Final Answer: Latest context the user wants to remember: "+input_data

def ReadContex(input_data):
    try:
        with open("context.txt", "r") as myfile:
            return myfile.read()
    except Exception as e:
        print(e)
        return "No context saved"

class TodoistTool:
    def __init__(self):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        todoist_api_token =  os.getenv('TODOIST_API_TOKEN')
        # get todoist tasks due today

        tasks_due_today = get_tasks_due_today(todoist_api_token)

        # create a chromaba database in langchain with persisted embeddings
        self.chromaba = Chroma(embedding_function=embeddings, persist_directory="models/chromaba.db")

        # for each todoist task, add it to the database if it doesn't exist
        for task in tasks_due_today:
            document = Document(page_content=json.dumps(task),
            #metadata looks like {'id': '7490784365', 'assigner_id': None, 'assignee_id': None, 'project_id': '2253901224', 'section_id': None, 'parent_id': None, 'order': 345, 'content': 'Order Christmas presents', 'description': '', 'is_completed': False, 'labels': [], 'priority': 1, 'comment_count': 0, 'creator_id': '590909', 'created_at': '2023-12-10T23:32:39.032517Z', 'due': {'date': '2023-12-15', 'string': 'Dec 15', 'lang': 'en', 'is_recurring': False}, 'url': 'https://todoist.com/showTask?id=7490784365', 'duration': None}
            metadata={
                "id": task["id"],
                "project_id": task["project_id"],
                "priority": task["priority"],
                "created_at": task["created_at"],
                "due": task["due"]["string"],
            },
            )
            id = task["id"]
            self.chromaba.add_documents([document],ids=[id],)
                
        # create a retriever from the chromaba database

        metadata_field_info = [
            AttributeInfo(
                name="id",
                description="The unique ID of the todoist task",
                type="integer",
            ),
            AttributeInfo(
                name="project_id",
                description="The id of the project the task belongs to",
                type="integer",
            ),
            AttributeInfo(
                name="priority",
                description="The priority of the task",
                type="string",
            ),
            AttributeInfo(
                name="created_at",
                description="the date the task was created",
                type="string",
            ),
            AttributeInfo(
                name="due",
                description="The due date of the task",
                type="string",
            ),
        ]
        document_content_description = "A single task from todoist"
        self.retriever = SelfQueryRetriever.from_llm(
            get_llm(),
            self.chromaba,
            document_content_description,
            metadata_field_info,
        )
        # self.retriever = self.chromaba.as_retriever(
        #     search_type="mmr",  # Also test "similarity"
        #     search_kwargs={"k": 8},
        # )
            

    def process(self, input_data):  
        llm = get_llm()
        
        memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", return_messages=True
        )
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=self.retriever, memory=memory,verbose=True)
        return qa.run(input_data)
