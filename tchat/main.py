from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

load_dotenv()

chat = ChatOpenAI()
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">> ")

    print(f"You entered: {content}")

    result = chain({"content": content})

    print(result["text"])
