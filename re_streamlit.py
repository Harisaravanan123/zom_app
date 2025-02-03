import folium.map
import streamlit as st
import pickle
import folium
from streamlit_folium import st_folium

import google.generativeai as genai

import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
connection=psycopg2.connect(
    host=os.getenv('host'),
    database=os.getenv('database'),
    port=os.getenv('port'),
    user= os.getenv('user'),
    password=os.getenv('password')
)  
connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
writer=connection.cursor()
writer.execute(''' SELECT * from "Res_data" ''')
tables=writer.fetchall()
column_names=[desc[0] for desc in writer.description]
df1=pd.DataFrame(tables,columns=column_names)  

with open('minikmeans.pkl','rb')as model_file:
    model=pickle.load(model_file)
with open('encoder.pkl','rb')as encoder_file:
    ohe=pickle.load(encoder_file)    
with open('pca.pkl','rb')as pca_file:
    pca=pickle.load(pca_file)  

st.set_page_config(page_title='Restaurant Recommendation System and Chefbot',page_icon='robot_face',layout='wide')


page=st.sidebar.radio("select a page:",("Restaurant Recommendation","ChefbotGPT"))  
if page=="Restaurant Recommendation":



# set the title
    title=st.title("RESTAURANT RECOMMENDATION SYSTEM")
    
    st.sidebar.header("Filter Options")
    df1['Cuisines'] = df1['Cuisines'].str.split(',')
    df1_exploed = df1.explode('Cuisines')
    cuisines_encoded = ohe.fit_transform(df1_exploed[['Cuisines']])
    cuisines_encoded_df = pd.DataFrame(cuisines_encoded,columns=ohe.get_feature_names_out(['Cuisines']))
    final_output = pd.concat([df1_exploed.reset_index(drop=True),cuisines_encoded_df.reset_index(drop=True)],axis=1)
    x=final_output[cuisines_encoded_df.columns]
    x_reduced = pca.fit_transform(x)
    final_output['Cluster'] = model.fit_predict(x_reduced)
    Cuisines = final_output['Cuisines'].unique()

    selected_Cuisines=st.sidebar.selectbox("SELECT THE CUISINES:",options=Cuisines)
    
    def get_recommendation(selected_cuisines):
        filtered_df=final_output[final_output["Cuisines"]==selected_Cuisines ]
        if filtered_df.empty:
            return pd.DataFrame()
        
        restaurant_recommendation=filtered_df[filtered_df['Cluster']==final_output['Cluster'][0]]
        restaurant_recommendation=restaurant_recommendation.sort_values(by='Rating', ascending=False)
        restaurant_recommendation=restaurant_recommendation.drop_duplicates(subset='Name')
        return restaurant_recommendation
    if selected_Cuisines:
        
        recommendations=get_recommendation(selected_Cuisines)   
        if not recommendations.empty:
            st.subheader("**RECOMMENDED_RESTAURANTS**")  
            st.dataframe(recommendations[['Name','Location','City','Rating']])
            st.subheader("**RESTAURANT LOCATION**")
            m=folium.Map(location=[recommendations['latitude'].mean(),recommendations['longitude'].mean()])
            for idx,row in recommendations.iterrows():
                folium.Marker(
                    location=[row['latitude'],row['longitude']],
                    popup=f"{row['Name']}  -  {row['Location']}",
                    icon=folium.Icon(color='blue')

                ).add_to(m)
            st_folium(m,width=1000,height=700)    
        else:
            st.write('no restaurants recommended')

# create a  chatbot  for cooking instructions only:
# st.sidebar.radio("chatbot")
if page=="ChefbotGPT":
    
    st.title("cooking instruction chatbot")
    
    st.header("Ask me anything about cooking")
    genai.configure(api_key=os.getenv('api_key'))
    model=genai.GenerativeModel("gemini-1.5-flash")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    user_input=st.text_input("Enter your cooking related questions:")
    if user_input:
        chat=model.start_chat(
            history=[
                {"role":"user",'parts':'hello'},
                {"role":"model","parts":"Give answers only for cooking related questions"},
            ]
        )
        response=chat.send_message(user_input)
        if response and response.candidates:
            bot_response=response.candidates[0].content.parts[0].text
        else:
            bot_response="Sorry I could not able to generate a response"    
        st.session_state.chat_history.append({'user':user_input,'bot':bot_response})
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write(f"**You:**{chat['user']}")  
            st.write(f"**bot:**{chat['bot']}")  