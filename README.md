CHEFMATE : RESTAURANT RECOMMENDATION SYSTEM AND COOKING ASSISTANT AI.

* The main objective of the project is to create a datascience clustering model to give restaurant recommendation based on cuisines and built an ai only for cooking role.

TOOLS

* PYTHON
* PANDAS,NUMPY
* SCIKIT-LEARN(KMEANS CLUSTERING,LABEL ENCODER,PCA)
* PSYCOPG2(POSTGRESQL)
* AWS(S3 BUKET,RDS,EC2 INSTANCE)
* STREAMLIT
  
DATA PREPROCESSING

* First push the raw restuarant data wjich was in dictionary format into aws s3 bucket.
* After that pull the data from s3 bucket to start doing preprocessing of the data .
* First we read the json format file and convert it into respective dataframes. There are totally 5 dataframes , I got then i concat into a single dataframe.
* Find null values in the column of the dataframe and remove it .
* Orgnaize the columns into their respective datatypes.
* Store the preprocessed data in the amazon RDS.
  
MACHINE LEARNING

* Read the data from amazon rds and used it for machine learning.
* I did onehotencoding for my cuisine column in my dataframe .
* After that I did PCA(Principal Component Analysis) for reduce the dimensionality of the dataset.
* I found the silhouette score in the seventh cluster(0.3580)
* I used minikmeans algorithm to train my dataset with n_clusters = 7.
* Then i pickle the encoder,pca and model file.
  
STREAMLIT

* First I read the data from rds with the use of psycopg2 .
* Then make my streamlit app called RESTAURANT RECOMMENDATION SYSTEM WITH INTEGRATED CHATBOT
* In the first page , I make a restaurant recommendation by selecting the cuisines,we got the restaurant in basis of top ratings with the name,location and city.
* I also integrated streamlit folium to show the location in the map with the latitiude and longitude of the restaurants.
* I named the integrated chatbot called as CHEFBOT.Which i located in the second page.
* In the chefbot, I used the google geminiai model to make the chatbot.
* Especially in the role ,i change it into cooking instructions, So i got everything related to cooking and food only.
  
DEPLOYMENT

* I created an EC2 INSTANCE WITH OS as ubuntu.
* After that I hosted my streamlit application in that ubuntu machine.
