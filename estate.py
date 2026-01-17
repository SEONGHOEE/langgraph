# %% [markdown]
# 병렬 처리를 위한 효율 개선
# - 특정 로직이 주어졌을 때, 두 개의 브랜치(노드 및 서브그래프)로 양분
# - 기존 방식에는 하나의 분기로 움직이는 시퀀스
# - 그러나, 해당 방식은 양분하는 병렬 처리 방식으로 시간 절약

# %% [markdown]
# <img src="/Users/seonghoe/인프런_랭그래프/lecture./langgraph/병렬처리_도식화.png">

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    answer: str # 세율
    # 우리에게 필요한 노드는 아래와 같음. (+ 이들에 대한 State 정의)
    tax_base_equation: str  # 과세표준 게산 수식
    tax_deduction: str  # 공제액 
    market_ratio: str   # 공정시장가액비율
    tax_base: str   # 과세표준 계산

graph_builder = StateGraph(AgentState)

# %% [markdown]
# 환경변수 설정 및 STATE 정의 완료

# %% [markdown]
# 과세표준 계산 수식 노드 생성 (→ 벡터 DB 내 존재)

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name='real_estate_tax',
    persist_directory='./real_estate_tax_collection'
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# %%
query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'

# %%
from langchain_openai import ChatOpenAI
from langchain import hub

rag_prompt = hub.pull('rlm/rag-prompt')
llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# 리트리빙을 위한 체인
tax_base_retrieval_chain = (
    {'context': retriever, 'question': RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ('system', '사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 리턴해주세요.'),
    ('human', '{tax_base_equation_information}')
])

# 결과를 수식으로 변환히기 위한 체인
tax_base_equation_chain = (
    {'tax_base_equation_information': RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()

)   

tax_base_chain = {'tax_base_equation_information': tax_base_retrieval_chain} | tax_base_equation_chain
 
def get_tax_base_equation(state: AgentState): 
    tax_base_equation_question = '주택에 대한 종합부동산세 게산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {'tax_base_equation': tax_base_equation}

# %% [markdown]
# RunnableMap({"context": retriever, "question": RunnablePassthrough()})
# - context = retriever.invoke(input)
# - question = input
# - RunnablePassthrough = '입력을 가공하지 않고 그대로 다음 단계로 넘기기 위한 장치'
# %% [markdown]
# 공제액 노드 생성 

# %%
tax_deduction_chain = (
    {'context': retriever, 'question': RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState): 
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 공제금액을 알려주세요.'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {'tax_deduction':  tax_deduction}

# %% [markdown]
# 공정시장가액비율 노드 생성 (→웹 서치)

# %%
from langchain_community.tools import TavilySearchResults
from datetime import date

tavily_search_tool = TavilySearchResults(
    max_results=7,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ('system', f'아래 정보를 기반으로 사용자의 질문에 답변해주세요\n\nContext:\n{{context}}'),
    ('human', '{query}')
])

def get_market_ratio(state: AgentState):
    query = f'오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇 % 인가요?'
    context = tavily_search_tool.invoke(query)
    tax_market_ratio_chain = (
        tax_market_ratio_prompt
        | llm 
        | StrOutputParser()
    )
    market_ratio = tax_market_ratio_chain.invoke({'context': context, 'query': query})
    return {'market_ratio': market_ratio}

# %%

# %% [markdown]
# 과세 표준 계산

# %%
from langchain_core.prompts import PromptTemplate

tax_base_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"""
주어진 내용을 기반으로 과세표준을 계산해주세요

과세표준 계산 공식: {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}
주의사항:
- LaTeX, 수식 기호 사용 금지
- 순수한 일반 텍스트로만 출력
- 특수기호 없이 숫자와 일반 괄호만 사용
         """),
        ('human', '사용자 주택 공시가격 정보: {query}')
    ]
)

def calculate_tax_base(state: AgentState):
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']
    tax_base_calculation_chain = (
        tax_base_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_base = tax_base_calculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })
    return{'tax_base': tax_base}
# %%

# %% [markdown]
# 세율 계산

# %% [markdown]
# <img src="/Users/seonghoe/인프런_랭그래프/lecture./langgraph/세율계산_도식화.png">

# %%
tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요.
종합부동산세 세율:{context}'''),
    
    ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요.
과세표준: {tax_base}
주택 수: {query}''')
])

def calculate_tax_rate(state: AgentState):
    query = state['query']
    tax_base = state['tax_base']
    context = retriever.invoke(query)
    tax_rate_chain = (
        tax_rate_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({
        'context': context,
        'tax_base': tax_base,
        'query': query
    })
    return {'answer': tax_rate}
# %% [markdown]
# 다음으로, 그래프 만들고 노드 추가 및 엣징

# %%
graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

# %%
graph = graph_builder.compile()
