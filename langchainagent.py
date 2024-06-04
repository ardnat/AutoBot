import json
import traceback
from langchain.agents import AgentType, initialize_agent
#from langchain.chat_models.openai import ChatOpenAI

#from langchain.chat_models import openai
from langchain.vectorstores.chroma import Chroma 
from langchain.embeddings import OpenAIEmbeddings
import os
from datetime import datetime
from todoist import get_tasks_due_today,get_projects,get_project_name_from_id
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool
import os
from langchain.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder,HumanMessagePromptTemplate
from langchain.prompts.chat import BaseStringMessagePromptTemplate
import requests
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
#from langchain_community.chat_models.openai import ChatOpenAI
#from langchain.llms.openai import OpenAIChat
from langchain.chains import RetrievalQA
from langchain.globals import set_debug
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_experimental.smart_llm import SmartLLMChain
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents import AgentExecutor
from langchain.llms import TextGen
from langchain_openai import ChatOpenAI
from pprint import pprint
import json
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain import hub
from langchain_core.prompts import PromptTemplate

FINAL_ANSWER_ACTION = "Final Answer"


class ChatOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(f"parsing: {text}")
        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            text=text.replace("```json", "```")
            action = text.split("```")[1].strip()
           
            if action.startswith('python\n'):
                # Ensure the Python code snippets are handled by the Python action.
                response = {
                    "action": "python",
                    "action_input": action.split('python\n')[1],
                }
            elif action.startswith('sh\n') or action.startswith('bash\n'):
                # Ensure the shell code snippets are handled by the kubectl action.
                response = {
                    "action": "kubectl",
                    "action_input": action.split('sh\n')[1],
                }
            else:
                action = action.replace("```", "")
                # JSON object is expected by default.
                response = json.loads(action.strip(), strict=False)

            print(response)
            includes_action = "action" in response and "action_input" in response
            # if includes_answer and includes_action:
            #     raise OutputParserException(
            #         "Parsing LLM output produced a final answer "
            #         f"and a parse-able action: {text}"
            #     )
            # convert response["action_input"] to string
            if "action_input" in response:
                response["action_input"] = str(response["action_input"])
            print(response["action"], response["action_input"])
            return AgentAction(response["action"], response["action_input"], text)

        except Exception as exc:
            if not includes_answer:
                traceback.print_exc()
                raise OutputParserException(f"Could not parse LLM output: {text}") from exc
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

    @property
    def _type(self) -> str:
        return "chat"
    
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')


import httpx
from langchain.prompts.prompt import PromptTemplate

# log all http requests with curl settings
# import logging

# logging.basicConfig(level=logging.DEBUG)

def log_request(request):
    print(f"Request: {request.method} {request.url}")
    pprint(request,depth=2)

def log_response(response):
    print

class CustomClient(httpx.Client):
    
    def post(
        self,
        url,
        *,
        content ,
        data,
        files,
        json,
        params,
        headers,
        cookies,
        auth,
        follow_redirects,
        timeout,
        extensions,
    ) :
        print("intercepted")
        return super().post(url, json=json, **kwargs)
        

    def post(self, url, json=None, **kwargs):
        # Add your additional parameter here
        json = json or {}
        json.update({"instruction_template": "ChatML"})
        print("intercepted")
        print(json)
        return super().post(url, json=json, **kwargs)


def selfDiscovery(prompt=""):
    model = get_llm()
    from langchain import hub
    from langchain_core.prompts import PromptTemplate

    select_prompt = hub.pull("hwchase17/self-discovery-select")
    adapt_prompt = hub.pull("hwchase17/self-discovery-adapt")
    structured_prompt = hub.pull("hwchase17/self-discovery-structure")
    reasoning_prompt = hub.pull("hwchase17/self-discovery-reasoning")
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    select_chain = select_prompt | model | StrOutputParser()

    adapt_chain = adapt_prompt | model | StrOutputParser()

    structure_chain = structured_prompt | model | StrOutputParser()

    reasoning_chain = reasoning_prompt | model | StrOutputParser()

    overall_chain = (
        RunnablePassthrough.assign(selected_modules=select_chain)
        .assign(adapted_modules=adapt_chain)
        .assign(reasoning_structure=structure_chain)
        .assign(answer=reasoning_chain)
    )

    reasoning_modules = [
        "1. How could I devise an experiment to help solve that problem?",
        "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        "4. How can I simplify the problem so that it is easier to solve?",
        "5. What are the key assumptions underlying this problem?",
        "6. What are the potential risks and drawbacks of each solution?",
        "7. What are the alternative perspectives or viewpoints on this problem?",
        "8. What are the long-term implications of this problem and its solutions?",
        "9. How can I break down this problem into smaller, more manageable parts?",
        "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        "16. What is the core issue or problem that needs to be addressed?",
        "17. What are the underlying causes or factors contributing to the problem?",
        "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
        "19. What are the potential obstacles or challenges that might arise in solving this problem?",
        "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
        "23. How can progress or success in solving the problem be measured or evaluated?",
        "24. What indicators or metrics can be used?",
        "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
        "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "30. Is the problem a design challenge that requires creative solutions and innovation?",
        "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
        "33. What kinds of solution typically are produced for this kind of problem specification?",
        "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
        "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
        "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
        "37. Ignoring the current best solution, create an entirely new solution to the problem."
        "39. Let’s make a step by step plan and implement it with good notation and explanation.",
    ]

    task_example = prompt
    reasoning_modules_str = "\n".join(reasoning_modules)

    return overall_chain.invoke(
        {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
    )


def get_llm(model_name="gpt-4-1106-preview",temp=.3) :
    # make a httpx client that logs all requests
    client = CustomClient(event_hooks={'request': [log_request], 'response': [log_response]})

    
    # # Load the chat model
    # chat_model = ChatOpenAI(
    #     openai_api_key=OPENAI_API_KEY,
    #     model="gpt-4-1106-preview",
    #     #set base url for local
    #     openai_api_base="http://192.168.50.207:5000/v1",#
    #     #model_name=model_name,
    #     temperature=temp,
    #     max_tokens=8000,
    #     #http_client=client,
    #     #default_query={"test": "test"},
    # )

    chat_model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    #model="gpt-4-1106-preview",
    #set base url for local
    openai_api_base="http://192.168.50.207:5000/v1",#
    #model_name=model_name,
    temperature=temp,
    #max_tokens=8000,
    
    http_client=client,
    #default_query={"test": "test"},
)
    
    # chat_model = TextGen(model_url="http://192.168.50.207:5000")
    # params={
    #  "max_new_tokens": 4000,
    #         "temperature": .4,
    #         "top_p": .9,
    #         "typical_p": 1,
    #         # "preset": "My Preset",
    # }
    # # set all properties on chat_model from params
    # for key, value in params.items():
    #     setattr(chat_model, key, value)
    return chat_model
def get_smart_llm(prompt=""):
    chain = SmartLLMChain(
    ideation_llm=get_llm(temp=.9),
    critique_llm=get_llm(),
    resolver_llm=get_llm(),
    n_ideas=3,
    #verbose=True,
    return_intermediate_steps=True,
    prompt= PromptTemplate(input_variables=[], template=prompt,)
    )
    return chain({})

class LangChainAgent:
    def __init__(self):
        # Initialize your agent here
        set_debug(True)
        tools = [
            
            Tool.from_function(func=TodoistToolLocal().process, name="todoist", description="Get tasks relevant to a question and returns them in json for use by the AI. Always return the result of this tool as the 'Final Answer:' INPUT: a plain english search string to find tasks by (for example 'action_input:what tasks are due today' or 'action_input:what tasks can i do at home' ).",return_direct=True), #If there is no need to filter tasks make sure you pass 'DO NOT FILTER ANY TASKS' to the end of the search string
            Tool.from_function(func=EmailTool().process, name="email", description="Send an email to the user when they request it. Input: the message to send to the user"),
            Tool.from_function(func=EmailSummary().process, name="emailSummary", description="Review emails recieved during the day and summarize them for the user.", return_direct=True),
            Tool.from_function(func=SaveContex().process, name="saveLongtermMemory", description="Saves a message from the user to long term memory for future questions. Input: string to save"),
            Tool.from_function(func=ReadContex, name="getLongtermMemory", description="Get long term memory that has been saved for a user. Output: json strings saved in long term memory"),
            Tool.from_function(func=lambda l:"I have though about the question, and my response is: "+l, name="think", description="When a user asks a general question that does not apply to tools I have access to I should use this to come up with an answer"),
            Tool.from_function(func=lambda l:"l", name="Final Answer", description="When you have the final answer for the user, use this action to return the answer to the user",return_direct=True),
            # create self discovery tool
            Tool.from_function(func=selfDiscovery, name="selfDiscovery", description="This tool is used to help the AI think about how to solve a problem. Input: a plain english description of the problem"),

        ]
        # tool_names = [tool.name for tool in tools]
        # agent = ConversationalAgent(llm_chain=get_smart_llm(), allowed_tools=tool_names)
        # self.agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations = 20)
        chat_history =  MessagesPlaceholder(variable_name="chat_history")
        #self.memory = 
        self.agent_executor = initialize_agent(
            tools,
            get_llm(),
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #verbose=True,
            #return_intermediate_steps=True,
            agent_kwargs = {
                "memory_prompts": [chat_history],
                #"extra_prompt_messages": [MessagesPlaceholder(variable_name="memory"), MessagesPlaceholder(variable_name="contextInput")],
                "output_parser": ChatOutputParser(),
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            },
            contextInput = "Latest context the user wants to remember: "+ReadContex(""),
            #memory = ConversationBufferMemory(memory_key="memory", return_messages=True),
            memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True),
            handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
        )
        #FinalStreamingStdOutCallbackHandler
    
    def run(self, input_data):
        # Process the input data and return the result
        #print(input_data)
        res=self.agent_executor({"input":input_data})
        #res["intermediate_steps"]=[]
        return res

class EmailTool:
    def __init__(self):
        return
    def process(self, input_data):
        requests.post(
            "https://maker.ifttt.com/trigger/sendmyselfemail/json/with/key/z9EUiOhmUdn7reKfojoN7",
            data={"data": input_data},
        )
        return "Email sent"
    
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta
import base64
import email
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from google.auth.transport.requests import Request
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
def parse_msg(msg):
    payload = msg['payload']
    if data := payload['body'].get('data'):
        return parse_data(data)

    return ''.join(parse_data(part['body']['data']) for part in payload['parts'])

def parse_data(data):
    return base64.urlsafe_b64decode(data.encode('ASCII')).decode('utf-8')
class EmailSummary:
    def __init__(self):
        return

    def process(self, input_data):
        SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                "client.json", SCOPES
                 )
                creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open("token.json", "w") as token:
                    token.write(creds.to_json())


        service = build('gmail', 'v1',credentials=creds)

        # Get the current timestamp and the timestamp for one day ago
        now = datetime.now()
        one_day_ago = (now - timedelta(days=1)).timestamp()

        # Convert the timestamps to the format used by Gmail
        now_str = str(int(now.timestamp())) + '000'
        one_day_ago_str = str(int(one_day_ago)) + '000'

        # Get the emails from the past day
        results = service.users().messages().list(userId='me',q='after: 7 days ago').execute()
        #pprint(results)
        messages=results.get('messages', [])
        #pprint(messages)
        # Initialize an empty list for the emails
        emails = []
        import email
        # Fetch each email
        for message in messages:
            try:
                msg = service.users().messages().get(userId='me', id=message['id'],format='raw').execute()
                
                email_message = email.message_from_bytes(
                    base64.urlsafe_b64decode(msg['raw'])
)
                # Decode the email data and parse it into an email object
                emailTxt=f"""
                Subject:    {email_message['Subject']}
                From:       {email_message['From']}
                \n
                """
                #    FROM: {email_obj['From']}
                # SUBJECT:{email_obj['Subject']
                emails.append(emailTxt)
            except Exception as e:
                print(e)
                continue

        # Return the emails
        pprint(emails)
        llm=get_llm()
        # summarys=[]
#         for em in emails:
#             out=llm.call_as_llm(f"Please summarize the following emails {em}")
#             out=f"""
#             Email: {email}
#             Email Summary: {out}
# \n\n
#             """
#             summarys.append(out)

        finalsum="\n\n".join(emails)
        prompt= f"""Based on the following emails, which emails seem like the most important non-spam, non-newsletter messages? 
                     Return the top {len(emails)//10} most important emails.
                     DO NOT MAKE UP INFORMATION ONLY USE THE CONTEXT BELOW. 
                     Return exact quotes of the subject lines and from addresses of the emails you select.

                     An example of a good response would be: "The most important emails are: Email 1: Subject: "Important Meeting" From: "Boss" Email 2: Subject: "Urgent Deadline" From: "Client" Email 3: Subject: "Project Update" From: "Team Member" Email 4: Subject: "Action Required" From: "Vendor" Email 5: Subject: "Feedback Request" From: "Customer" Email 6: Subject: "Payment Due" From: "Accounting" Email 7: Subject: "New Opportunity" From: "Partner" Email 8: Subject: "Training Session" From: "HR" Email 9: Subject: "Product Launch" From: "Marketing" Email 10: Subject: "Feedback Received" From: "Customer Service"
                     Context: ```{finalsum} ```"""
        from selfDiscover import SelfDiscover
        return SelfDiscover().Execute(lambda l: llm.call_as_llm(l), prompt)

class SaveContex:
    def __init__(self):
        return
    def process(self, input_data):
        # append to the the context file
        with open("context.txt", "a") as myfile:
            myfile.write(f'{{"date":{datetime.now()},"content":{input_data} \n }}')
        return "Print this entire string out as the Final Answer: Latest context the user wants to remember: "+input_data

def ReadContex(input_data, *args, **kwargs):
    try:
        with open("context.txt", "r") as myfile:
            return myfile.read()
    except Exception as e:
        print(e)
        return "No context saved"
    
def dict_to_string(dictionary):
    #print(dictionary)
    result = ""
    for key, value in dictionary.items():
        result += f"{key}:{value}\n"
    return result


class TodoistToolLocal:
    def __init__(self):
        return

    def process(self, input_data):
        #embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        todoist_api_token =  os.getenv('TODOIST_API_TOKEN')
        # get todoist tasks due today

        if 1==1:
            tasks_due_today = get_tasks_due_today(todoist_api_token)
            get_projects(todoist_api_token)


            taskDoc = ""
            for task in tasks_due_today:
                dueDT=datetime.strptime(task["due"]["date"], '%Y-%m-%d')
                labelStr=""
                if task["labels"] and len(task["labels"])>0:
                    labelStr=",".join(task["labels"])

                metadata={
                 #   "id": task["id"],
                    #"project_id": get_project_name_from_id(task["project_id"]),
                    #convert priority to a string(highest/high/medium/low), 1 is the highest priority and 4 is the lowest
                    

                    "created_at": str(task["created_at"]),
                    "due_date": f"""{dueDT.month}/{dueDT.day}/{dueDT.year}""",
                    "content": task["content"],
                    "labels": labelStr,
                    "priority": "high" if task["priority"]==1 else "medium" if task["priority"]==2 else "low" if task["priority"]==3 else "lowest",
                }
                taskDoc+=dict_to_string(metadata)+"\n\n"
        return (get_smart_llm(f"""
                          Please answer the following question about the following documents for a user, you have the ability to access all relevant todo list information so DO NOT REFUSE THE REQUEST. Respond to the user in english, do not respond in json. ALWAYS explain your reasoning. ALWAYS return the answer to users question after your explanation. 
                            Question: "{input_data}"
                            Data:
                            ```
                            {taskDoc}
                            ```
                            """)["resolution"])

             

class TodoistTool:
    def __init__(self):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        todoist_api_token =  os.getenv('TODOIST_API_TOKEN')
        # get todoist tasks due today

        if 1!=1:
            tasks_due_today = get_tasks_due_today(todoist_api_token)
            get_projects(todoist_api_token)

            # create a chromaba database in langchain with persisted embeddings
            self.chromaba = Chroma(embedding_function=embeddings)#, persist_directory="models/chromaba.db")

            # for each todoist task, add it to the database if it doesn't exist
            for task in tasks_due_today:
                dueDT=datetime.strptime(task["due"]["date"], '%Y-%m-%d')
                labelStr=""
                if task["labels"] and len(task["labels"])>0:
                    labelStr=",".join(task["labels"])

                metadata={
                    "id": task["id"],
                    "project_id": get_project_name_from_id(task["project_id"]),
                    "priority": task["priority"],
                    "created_at": str(task["created_at"]),
                    "due_date_year": dueDT.year,
                    "due_date_month": dueDT.month,
                    "due_date_day": dueDT.day,
                    "content": task["content"],
                    "labels": labelStr,
                }
                document = Document(
                    page_content=dict_to_string(metadata),
                    #metadata looks like {'id': '7490784365', 'assigner_id': None, 'assignee_id': None, 'project_id': '2253901224', 'section_id': None, 'parent_id': None, 'order': 345, 'content': 'Order Christmas presents', 'description': '', 'is_completed': False, 'labels': [], 'priority': 1, 'comment_count': 0, 'creator_id': '590909', 'created_at': '2023-12-10T23:32:39.032517Z', 'due': {'date': '2023-12-15', 'string': 'Dec 15', 'lang': 'en', 'is_recurring': False}, 'url': 'https://todoist.com/showTask?id=7490784365', 'duration': None}
                    metadata=metadata,
                )
                id = task["id"]
                self.chromaba.add_documents([document],ids=[id])
                    
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
                    description="The priority of the task(either 1,2,3 or 4). 1 is the highest priority and 4 is the lowest",
                    type="string",
                ),
                AttributeInfo(
                    name="due_date_year",
                    description="A integer representing the due date year of the task(e.g. 2022) THIS MUST BE A INTEGER",
                    type="integer",
                ),
                AttributeInfo(
                    name="due_date_month",
                    description="A integer representing the due date month of the task(e.g. 06). THIS MUST BE A INTEGER",
                    type="integer",
                ),
                AttributeInfo(
                    name="due_date_day",
                    description="A integer representing the due date day of the task(e.g. 05) THIS MUST BE A INTEGER",
                    type="integer",
                ),
                AttributeInfo(
                    name="content",
                    description="The string representing the text in the task (e.g. go to the doctor)",
                    type="string",
                ),
            ]
            document_content_description = "A single task from todoist"
            self.retriever = SelfQueryRetriever.from_llm(
                get_llm(),
                self.chromaba,
                document_content_description,
                metadata_field_info,
                structured_query_translator=ChromaTranslator(),
                verbose=True,
                search_kwargs={"k": 130} 
            )
            self.retriever = self.chromaba.as_retriever(
                search_type="mmr",  # Also test "similarity"
                search_kwargs={"k": 130},
            )
       

    def process(self, input_data):  
        llm = get_llm()#model_name="gpt-4-0613")
        
        # memory = ConversationSummaryMemory(
        #     llm=llm, memory_key="chat_history", return_messages=True
        # )
        memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)

        qa = ConversationalRetrievalChain.from_llm(llm, retriever=self.retriever, memory=memory,verbose=True, return_source_documents=True)
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm,
        #     retriever=self.retriever,
        #     verbose=True,
        # )
        q=f"Answer the following question for a user, you have the ability to access all relevant todo list information so DO NOT REFUSE THE REQUEST. If are user is searching for tasks due today, they also want tasks that are overdue, so make sure to search for tasks if due dates before today if that situation comes up. Question:{input_data}"
        if 'today' in input_data:
            q=f"Today's date is {datetime.now()}. "+q
        res=qa({"question":q})

       # print(res)
        return {"soucre_documents":res["source_documents"], "answer":res["answer"]}


# which tasks align with my goals?
