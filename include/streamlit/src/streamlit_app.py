import os
import json
import pandas as pd
import streamlit as st
import logging
import sys

from snowflake.snowpark import Session, Window
from snowflake.snowpark import functions as F

import weaviate

from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

st.set_page_config(layout="wide")


def get_logger():
    """
    Get a logger for local logging.
    """
    logger = logging.getLogger("job-tutorial")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_login_token():
    """
    Read the login token supplied automatically by Snowflake. These tokens
    are short lived and should always be read right before creating any new
    connection.
    """
    with open("/snowflake/session/token", "r") as f:
        return f.read()

def get_connection_params():
    """
    Construct Snowflake connection params from environment variables.
    """
    if os.path.exists("/snowflake/session/token"):
        return {

        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "host": os.getenv("SNOWFLAKE_HOST"),
        "authenticator": "oauth",
        "token": get_login_token(),
        # "warehouse": SNOWFLAKE_WAREHOUSE,
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA")
        }
    else:
        snowflake_conn_params = json.loads(os.environ['SNOWFLAKE_CONN_PARAMS'])
        return {
        "account": snowflake_conn_params['extra']['account'],
        "user": snowflake_conn_params['login'],
        "password": snowflake_conn_params['password'],
        "role": snowflake_conn_params['extra'].get('role'),
        "warehouse": snowflake_conn_params['extra'].get('warehouse'),
        "database": snowflake_conn_params['extra'].get('database'),
        "schema": snowflake_conn_params.get('schema'),
        "region": snowflake_conn_params.get('region')
}

# logger = get_logger()
# logger.info()

if "snowpark_session" not in st.session_state:
  snowpark_session = Session.builder.configs(get_connection_params()).create()
  st.session_state['snowpark_session'] = snowpark_session
else:
  snowpark_session = st.session_state['snowpark_session']

if "weaviate_client" not in st.session_state:
    weaviate_client = weaviate.Client(
            url = os.environ['WEAVIATE_ENDPOINT_URL'].replace("\'",""), 
                additional_headers = {
                    "X-OpenAI-Api-Key": os.environ['OPENAI_APIKEY']
                }
            )
    st.session_state['weaviate_client'] = weaviate_client
else:
  weaviate_client = st.session_state['weaviate_client']    


#Setup data sources
attribution_df = snowpark_session.table('ATTRIBUTION_TOUCHES')
calls_df = snowpark_session.table('PRED_CUSTOMER_CALLS')
comments_df = snowpark_session.table('PRED_TWITTER_COMMENTS')

@st.cache_data
def ad_spend_df():
    return snowpark_session.table('PRES_AD_SPEND').to_pandas()
_ad_spend_df = ad_spend_df()

@st.cache_data
def clv_df(row_limit=10):
    df = snowpark_session.table('PRES_CLV').to_pandas()
    df.columns = [col.replace('_', ' ') for col in df.columns]
    df.set_index('CUSTOMER ID', drop=True, inplace=True)
    return df
_clv_df = clv_df()

@st.cache_data
def churn_df(row_limit=10):
    df = snowpark_session.table('PRES_CHURN').to_pandas()
    df.columns = [col.replace('_', ' ') for col in df.columns]

    return df
_churn_df = churn_df()

def aggrid_interactive_table(df: pd.DataFrame, height=400):
    options = GridOptionsBuilder.from_dataframe(
        df, 
        enableRowGroup=True, 
        enableValue=True, 
        enablePivot=True,
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        height=height,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

with st.container():              
    title_col, logo_col = st.columns([8,2])  
    with title_col:
        st.title("GroundTruth: Customer Analytics")
        st.subheader("Powered by Apache Airflow, Snowpark Containers and Streamlit")
    with logo_col:
        st.image('logo_small.png')

marketing_tab, sales_tab, product_tab = st.tabs(["Marketing", "Sales", "Product"])

with marketing_tab:
    with st.container():
        st.header("Return on Ad Spend")
        st.subheader("Digital Channel Revenue")

        medium_list = [i.UTM_MEDIUM for i in attribution_df.select(F.col('UTM_MEDIUM')).na.fill('NONE').distinct().collect()]
        
        st.bar_chart(_ad_spend_df, x="MEDIUM", y="REVENUE")

        st.subheader('Revenue by Ad Medium')
        st.table(attribution_df.select(['UTM_MEDIUM', 'UTM_SOURCE', 'REVENUE'])\
                                .pivot(F.col('UTM_MEDIUM'), medium_list)\
                                .sum('REVENUE')\
                                .rename('UTM_SOURCE', 'Source')
        )

with sales_tab:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header("Top Customers by Customer Lifetime Value")
            st.dataframe(_clv_df.style.background_gradient(cmap='Reds', subset=['SENTIMENT SCORE'], axis=None))

        with col2:
            st.header("Churned Customers")        
            selection_churn = aggrid_interactive_table(_churn_df, height=400)

    with st.container():

        if len(selection_churn['selected_rows']) > 0:
            selected_customer_id = selection_churn['selected_rows'][0]['CUSTOMER ID']
            
            selected_calls_df = calls_df.filter(F.col('CUSTOMER_ID') == selected_customer_id)\
                                        .drop(F.col('UUID'))\
                                        .sort(F.col('SENTIMENT'), ascending=False)\
                                        .to_pandas()
            selected_comments_df = comments_df.filter(F.col('CUSTOMER_ID') == selected_customer_id)\
                                            .drop(F.col('UUID'))\
                                            .sort(F.col('SENTIMENT'), ascending=False)\
                                            .limit(5)\
                                            .to_pandas()
            
            st.header('Customer Support Calls')
            selection_call = aggrid_interactive_table(selected_calls_df, height=200)

            if len(selection_call['selected_rows']) > 0:
                st.subheader('Call Transcription')
                st.write(selection_call['selected_rows'][0]['TRANSCRIPT'])
                st.subheader('Call Audio')
                audio_url = snowpark_session.sql(f"SELECT get_presigned_url(@call_stage, LIST_DIR_TABLE.RELATIVE_PATH) as presigned_url \
                                                   FROM DIRECTORY( @call_stage) \
                                                   WHERE RELATIVE_PATH = '{selection_call['selected_rows'][0]['RELATIVE_PATH']}';")\
                                            .collect()[0]\
                                            .as_dict()['PRESIGNED_URL']
                st.audio(audio_url, format='audio/wav') 

            st.header('Customer Social Media Comments')
            selection_comment = aggrid_interactive_table(selected_comments_df, height=200)
    
            if len(selection_comment['selected_rows']) > 0:
                st.subheader('Twitter comment text')
                st.write(selection_comment['selected_rows'][0]['REVIEW_TEXT'])

with product_tab:
    with st.container():
        st.header('Product Research')
        search_text = st.text_input('Customer Feedback Keyword Search', value="")

        if search_text:
            st.write("Showing vector search results for: "+search_text)

            nearText = {"concepts": [search_text]}

            col1, col2 = st.columns(2)
            with col1:

                comments_result = weaviate_client.query\
                                                 .get("CustomerComment", ["cUSTOMER_ID", "dATE", "rEVIEW_TEXT"])\
                                                 .with_additional(['id'])\
                                                 .with_near_text(nearText)\
                                                 .with_limit(3)\
                                                 .do()
                
                if comments_result.get('errors'):
                    for error in comments_result['errors']:
                        if 'no api key found' or 'remote client vectorize: failed with status: 401 error' in error['message']:
                            raise Exception('Cannot vectorize.  Check the OPENAI_API key as environment variable.')
                
                near_comments_df = pd.DataFrame(comments_result['data']['Get']['CustomerComment'])\
                        .rename(columns={'cUSTOMER_ID':'CUSTOMER ID', 'dATE':'DATE', 'rEVIEW_TEXT':'TEXT'})
                near_comments_df['UUID'] = near_comments_df['_additional'].apply(lambda x: x['id'])
                near_comments_df.drop('_additional', axis=1, inplace=True)

                st.subheader('Customer Twitter Comments')
                selection_comment = aggrid_interactive_table(near_comments_df, height=100)
                
                if len(selection_comment['selected_rows']) > 0:
                    st.write(selection_comment['selected_rows'][0]['TEXT'])

                    st.subheader('Named Entities Extracted: ')
                    result = weaviate_client.query\
                            .get("CustomerComment", ["rEVIEW_TEXT", 
                                                     "_additional {tokens ( properties: [\"rEVIEW_TEXT\"], limit: 1, certainty: 0.5) {entity property word certainty startPosition endPosition }}"])\
                            .with_where(
                                {
                                    'path': ["id"],
                                    'operator': 'Equal',
                                    'valueString': selection_comment['selected_rows'][0]['UUID']
                                }
                            )\
                            .do()
                    
                    # NER_string = ''
                    # tokens = result['data']['Get']['CustomerComment'][0]['_additional']['tokens']
                    # for token in range(len(tokens)):
                    #     NER_string = NER_string + tokens[token]['word'] + ' : ' + tokens[token]['entity'] + ', '

                    # st.write(NER_string)

            with col2:
                call_result = weaviate_client.query\
                                             .get("CustomerCall", ["cUSTOMER_ID", "tRANSCRIPT"])\
                                             .with_additional(['id'])\
                                             .with_near_text(nearText)\
                                             .with_limit(3)\
                                             .do()\
                
                near_calls_df = pd.DataFrame(call_result['data']['Get']['CustomerCall'])\
                        .rename(columns={'cUSTOMER_ID':'CUSTOMER ID', 'tRANSCRIPT':'TEXT'})
                near_calls_df['UUID'] = near_calls_df['_additional'].apply(lambda x: x['id'])
                near_calls_df.drop('_additional', axis=1, inplace=True)

                st.subheader('Customer Calls')
                selection_call = aggrid_interactive_table(near_calls_df, height=100)

                if len(selection_call['selected_rows']) > 0:
                    st.write(selection_call['selected_rows'][0]['TEXT'])

                    st.subheader('Named Entities Extracted: ')
                    result = weaviate_client.query\
                                .get("CustomerCall", 
                                     ["tRANSCRIPT", 
                                      "_additional {tokens ( properties: [\"tRANSCRIPT\"], limit: 1, certainty: 0.7) {entity property word certainty startPosition endPosition }}"])\
                                .with_where(
                                    {
                                        'path': ["id"],
                                        'operator': 'Equal',
                                        'valueString': selection_call['selected_rows'][0]['UUID']
                                    }
                                )\
                                .do()
                    
                    # NER_string = ''
                    # tokens = result['data']['Get']['CustomerCall'][0]['_additional']['tokens']
                    # for token in range(len(tokens)):
                    #     NER_string = NER_string + tokens[token]['word'] + ' : ' + tokens[token]['entity'] + ', '

                    # st.write(NER_string)
                    


    with st.container():
        st.header('QNA Search')
        search_question = st.text_area('Customer Feedback QNA Search', value="")

        if search_question:
            st.write("Showing QNA search results for:  "+search_question)

            col1, col2 = st.columns(2)
            with col1:
                col1.subheader('Twitter Comment Results')
                comment_ask = {
                    "question": search_question,
                    "properties": ["rEVIEW_TEXT"]
                }

                result = weaviate_client.query\
                            .get("CustomerComment", ["rEVIEW_TEXT", 
                                                     "_additional {answer {hasAnswer property result startPosition endPosition} }"])\
                            .with_ask(comment_ask)\
                            .with_limit(1)\
                            .do()
                
                if result.get('errors'):
                        for error in result['errors']:
                            if 'no api key found' or 'remote client vectorize: failed with status: 401 error' in error['message']:
                                raise Exception('Cannot vectorize.  Check the OPENAI_API key as environment variable.')

                if result['data']['Get']['CustomerComment']: 
                    st.write(result['data']['Get']['CustomerComment'][0]['rEVIEW_TEXT'])

            with col2:
                col2.subheader('Customer Call Results')
                call_ask = {
                    "question": search_question,
                    "properties": ["tRANSCRIPT"]
                }

                result = weaviate_client.query\
                            .get("CustomerCall", 
                                ["tRANSCRIPT", 
                                "_additional {answer {hasAnswer property result startPosition endPosition} }"])\
                            .with_ask(call_ask)\
                            .with_limit(1)\
                            .do()

                if result['data']['Get']['CustomerCall']: 
                    st.write(result['data']['Get']['CustomerCall'][0]['tRANSCRIPT'])
