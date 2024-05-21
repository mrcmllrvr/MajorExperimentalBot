__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import openai
import json
from openai import OpenAI
from streamlit_feedback import streamlit_feedback

import logging
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip

class ContextFilter(logging.Filter):
    def filter(self, record):
        record.user_ip = get_remote_ip()
        return super().filter(record)

def init_logging():
    # Make sure to instanciate the logger only once
    # otherwise, it will create a StreamHandler at every run
    # and duplicate the messages

    # create a custom logger
    logger = logging.getLogger("MajorTravelUAT")
    if logger.handlers:  # logger is already setup, don't setup again
        return
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # in the formatter, use the variable "user_ip"
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s [user_ip=%(user_ip)s] - %(message)s")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.addFilter(ContextFilter())
    handler.setFormatter(formatter)
    logger.addHandler(handler)

init_logging()

logger = logging.getLogger(f"MajorTravelUAT")

COHERE_KEY = st.secrets['COHERE_KEY']
openai_api_key = st.secrets['OPENAI_API_KEY']

with st.sidebar:
    # New elements for sidebar UI
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #CF287A;
        }
        .sidebar-img {
            width: 50%;
            height: auto;  /* Maintain aspect ratio */
            max-width: 150px;  /* Set a max width for the images */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar content using HTML for image
    st.markdown('<img src="major-white.png" class="sidebar-img">', unsafe_allow_html=True)
    StreamlitUser = st.text_input("Hi! May I know who is utilizing the tool?", key="StreamlitUser")
    
##### CONNECT TO DATABASE and OpenAI #####
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = 'chromadb_major_travel/'
HYPO_COLLECTION_NAME = "hypothetical_embeddings"
DOCS_COLLECTION_NAME = "document_embeddings"

def text_embedding(text):
    response = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return response["data"][0]["embedding"]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )

client = chromadb.PersistentClient(path = CHROMA_DATA_PATH)

document_collection = client.get_or_create_collection(
    name = DOCS_COLLECTION_NAME,
    embedding_function = openai_ef,
    metadata = {"hnsw:space" : "cosine"}
)

questions_collection = client.get_or_create_collection(
    name = HYPO_COLLECTION_NAME,
    embedding_function = openai_ef,
    metadata = {"hnsw:space" : "cosine"}
)

# Startup ChromaDB with Initial Query
qcol_init = questions_collection.query(
        query_texts = ["Initialize Collection"],
        n_results = 5,
        include=["documents","distances","metadatas"]
    )

logger.info(f"Succesfully Initialized Hypothetical Questions Database")

dcol_init = document_collection.query(
        query_texts = ["Initialize Collection"],
        n_results = 5,
        include=["documents","distances","metadatas"]
    )

logger.info(f"Succesfully Initialized Documents Database")

del qcol_init
del dcol_init

######################################


OpenAIClient = openai.OpenAI(
    api_key=openai_api_key,
)
###############################

#### DEFINE FUNCTION CALLS ##############
import cohere
co = cohere.Client(COHERE_KEY)
def get_relevant_question_context(query, 
                                  limit = 15, 
                                  include_document_in_retrieval = True, 
                                  priority_SOP = None):
    if not priority_SOP:
        relevant_questions = questions_collection.query(
            query_texts = [query],
            n_results = limit,
            include=["documents","distances","metadatas"]
        )
    else:
        relevant_questions = questions_collection.query(
            query_texts = [query],
            n_results = limit,
            include=["documents","distances","metadatas"],
            where = {'Filename': priority_SOP}
        )
    
    distance_threshold = 0.6
    questions = []
    metadatas = []
    for dist_lst, document_lst, meta_lst in list(zip(relevant_questions['distances'], relevant_questions['documents'], relevant_questions['metadatas'])):
        for dst, doc, meta in list(zip(dist_lst, document_lst, meta_lst)):
            if dst <= distance_threshold:
                if doc not in questions:
                    questions.append(doc) 
                    metadatas.append(meta)
                    
                    #os.write(1,b"Relevant Questions:\n")
                    #os.write(1,f"DISTANCE : {dst}\nCONTENT : {doc}".encode())
                
    if include_document_in_retrieval:
        docu_context = document_collection.query(
            query_texts = [query],
            n_results = limit,
            include=["documents","distances","metadatas"]
        )
    
        for docu_dst, doc_text, meta_text in list(zip(docu_context['distances'], docu_context['documents'], docu_context['metadatas'])):
            for dst, doc, meta in list(zip(docu_dst, doc_text, meta_text)):
                #os.write(1,b"\n\nSimilar Raw Documents:\n")
                #os.write(1,f"DISTANCE : {dst}\nCONTENT : {doc}".encode())
                if dst <= 0.8:
                    questions.append(doc)
                    metadatas.append(meta)

                    #os.write(1,b"\n\nRelevant Raw Documents:\n")
                    #os.write(1,f"DISTANCE : {dst}\nCONTENT : {doc}".encode())
                    
    if questions:
        index2doc = {doc : i for i,doc in enumerate(questions)}
        results = co.rerank(query=query, documents=questions, top_n=5, model='rerank-english-v3.0', return_documents=True)   
        questions = [str(r.document.text) for r in results.results] 
        questions_indexes = [index2doc[doc] for doc in questions]
        relevant_metadatas = [metadatas[i] for i in questions_indexes] 
        
        unique_metadatas = [dict(t) for t in {tuple(d.items()) for d in relevant_metadatas}]
        print(unique_metadatas)
        filenames = [f"{mt['Filename']}-{mt['Section Name']}" for mt in unique_metadatas]
        
        relevant_raw_documents = []
        for relevant_q_meta in unique_metadatas:
            meta_filter = {"$and": [{k : {"$eq" : v}} 
                                    for k,v in relevant_q_meta.items()
                                    if k != 'document_index']}
            rds = document_collection.get(where=meta_filter,
                                        include=['documents'])
            try:
                relevant_raw_documents.append(rds['documents'][0])
            except IndexError:
                pass
        relevant_raw_documents = list(set(relevant_raw_documents))
        doc2filename = {docu:file for docu, file in list(zip(relevant_raw_documents, filenames))}
        
        doc_rerank = co.rerank(query=query, documents=relevant_raw_documents, top_n=3, model='rerank-english-v3.0', return_documents=True)   
        reranked_documents = [str(r.document.text) for r in doc_rerank.results] 
        reranked_filenames = [doc2filename[rrd] for rrd in reranked_documents]
        
        FN_DOC = [f"CONTEXT_SOURCE_FILE:{file}\nCONTENT:{docu}\n" for file,docu in list(zip(reranked_filenames, reranked_documents))]
        context_data = "\n".join(FN_DOC)
        context_str = f"""
        You may use the following SOP Documents to answer the question:
        
        {context_data}
        """
        #os.write(1,f"Relevant Context\n\n{context_str}".encode())
        return context_str
        
    else:
        return "NO RELEVANT CONTEXT FOUND"

SIGNATURE_get_relevant_question_context = {
    "type" : "function",
    "function" : {
        "name" : "get_relevant_question_context",
        "description" : "Get related SOPs to use as context from ChromaDB",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "query" : {
                    "type" : "string",
                    "description" : "Query passed by the user to the chatbot"
                },
                "limit" : {
                    "type" : "integer",
                    "description" : "Total number of SOPs to retrieve from vector database"
                }
            },
            "required" : ["query"],
        }
    }
    
}

def get_AYNTK_documents(query, limit = 15, include_document_in_retrieval = True, priority_SOP = "AYNTK"):
    return get_relevant_question_context(query, limit = 15, include_document_in_retrieval = True, priority_SOP = "AYNTK")
    
SIGNATURE_get_AYNTK_context = {
    "type" : "function",
    "function" : {
        "name" : "get_AYNTK_documents",
        "description" : "First function to use if in need of context to answer a query. Finds answers within the AYNTK document.",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "query" : {
                    "type" : "string",
                    "description" : "Query passed by the user to the chatbot"
                },
                "limit" : {
                    "type" : "integer",
                    "description" : "Total number of SOPs to retrieve from vector database"
                }
            },
            "required" : ["query"],
        }
    }
    
}

tools = [SIGNATURE_get_relevant_question_context, SIGNATURE_get_AYNTK_context]

#######################################


###### ADD CUSTOM FUNCTIONS ###########
import tiktoken
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        try:
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += -1
        except:
            num_tokens += len(encoding.encode(str(message)))
    num_tokens += 2
    return num_tokens

############### PROMPTS ###############

system_prompt = """
Role : As a travel agent assistant for Major Travel, your role involves strictly adhering to the agency's standard operating procedures (SOPs) and assisting your coworkers with their queries about it.

Response Rules:
1. Utilize the provided context given to you in identifying which SOP is relevant to their query through document retrieval. 
2. Respond in a friendly and professional tone - much like a travel agent or bank assistant giving answers to questions asked of them.
3. If need be, perform multiple function calls to answer the question
4. If you deem necessary, try interpreting questions by making them less vague. Take for example an instance where the use of "who do we use" can also be understood as "Who does Major Travel use".

Retrictions:
1. Answer only with facts extracted from the context provided to you. Do not generate answers that don't use the sources provided to you.
2. Try to infer some details but not the extent that you are already using information not provided to you by the SOPs. Only do so for instances like infering acronyms and vague wording.
3. Avoid mentioning everything any information irrelevant to your coworkers' questions - try to be concise while remaining informative.

Guidelines for responses:
1. In the case that the prompt was too vague, ask clarifying questions.
2. If you think you have a similar context to the prompt - even if not exactly the same - ask if thats what they meant. 
3. If no similar context was found from your initial search, ask clarifying questions that would help you answer them better or help you identify what to look for.
4. If you still dont know the answer, respond by saying that you were unable to find a good answer but inform them which SOP document you found most similar.
5. In the instance that the question is incomprehensible, respond accordingly by saying that you only have knowledge about the SOPs.
6. If doing function calls, run get_AYNTK_documents first to check for relevant context. If no relevant context is found within AYNTK, thats the only time you can run get_relevant_question_context

Response Template:
<start of template>
Relevant Context found in {CONTEXT_SOURCE_FILE}\n
{PROMPT_RESPONSE}
</end of template>
"""
########################################
st.title("📝 Major Travel Chatbot UAT Platform")
if StreamlitUser:

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {'role' : 'system' , 'content' : system_prompt},
            {"role": "assistant", "content": "How can I help you? Leave feedback to help me improve!"}
        ]
        
    if "response" not in st.session_state:
        st.session_state["response"] = ''
        
    messages = st.session_state.messages
    for msg in messages:
        try:
            if msg['role'] in ['user', 'assistant']:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])
        except:
            pass
            #print(msg)
            
    # delete older completions to keep conversation under token limit
    while num_tokens_from_messages(messages) >= 8192*0.8:
        messages.pop(0)
        
    if prompt := st.chat_input(placeholder="What do you want to know about Major Travel's SOPs"):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        logger.info(f"From Prompt Cleaner of {StreamlitUser} - {prompt}")
        messages.append({"role" : "user" , "content" : prompt})
        

        with st.chat_message("assistant"):
            with st.spinner("Looking Up Answer 📖..."):
                # Get Response
                response = OpenAIClient.chat.completions.create(
                    messages=messages,
                    model="gpt-4",
                    temperature=0,
                    n=1,
                    seed = 82598,
                    tools = tools,
                    tool_choice = "auto"
                )
                
                response_message = response.choices[0].message
                logger.info(f"For {StreamlitUser} - {response_message}")
                tool_calls = response_message.tool_calls
                
                if tool_calls:
                    available_fxns = {
                        "get_relevant_question_context" : get_relevant_question_context,
                        "get_AYNTK_documents" : get_AYNTK_documents
                    }
                    
                    messages.append(response_message),
                    
                    for tool_call in tool_calls:
                        fxn_name = tool_call.function.name
                        fxn_to_call = available_fxns[fxn_name]
                        fxn_args = json.loads(tool_call.function.arguments)
                        fxn_response = fxn_to_call(
                            **fxn_args
                        )
                        
                        messages.append(
                            {
                                "tool_call_id" : tool_call.id,
                                "role" : "tool",
                                "name" : fxn_name,
                                "content" : fxn_response
                            }
                        )
                context_enhanced_response = OpenAIClient.chat.completions.create(
                    messages=messages,
                    model="gpt-4",
                    seed = 82598,
                    temperature=0,
                    n=1,
                )
                
                # Extract Answer
                answer = context_enhanced_response.choices[0].message.content
                st.session_state["response"] = answer
                messages.append({"role" : "assistant", "content" : st.session_state["response"]})
                logger.info(f"From Chatbot in response to User ({StreamlitUser}) - {st.session_state['response']}")
                if st.session_state["response"]:
                    st.markdown(st.session_state["response"])
            
    if st.session_state["response"]:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key = f"feedback_{len(messages)}"
        )
    
        feedback_score_map = {"👍": "Good", "👎": "bad"}
        if feedback:
            score = feedback_score_map.get(feedback["score"])
            if score is not None:
                feedback_str = f"{score} Answer : {feedback.get('text')}"
                logger.info(f"Feedback from {StreamlitUser}- {feedback_str}")
                
            st.toast("Feedback recorded!", icon="📝")
else:
    st.warning("Please input name on the sidebar first prior to proceeding with the UAT")
