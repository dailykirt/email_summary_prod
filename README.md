# Extractive Summarization Application
This is the code that can be hosted to explore the Enron email dataset using extractive summarization methods. 

# Web Interface
You can explore the Enron summaries [here](http://enron-emails.herokuapp.com/). 
Select an inbox then click on 'Display valid dates' to see the time range of emails you can summarize. Then click on one of the following buttons:

'Summaries': Produces extractive summaries for all the emails in a given time frame. 
'Emails': Displays every email. 
'Summaries and emails': Give summaries, but also show the email the extractive summary was found in. 

# Running Local

1. Clone this repository and install requirements: ‘pip install -r requirements.txt’
1. Download an email dataset. The Enron emails found [here](https://www.cs.cmu.edu/~enron/) works best for this application. Unzip it into the data directory in the ‘data_processing’ folder. The full path would be: ‘./data_processing/data/maildir’
1. At the root level, create a ‘config.py’ file. Have the line ‘class Config(object):’ then set the following variables
   1. SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/local_table’
   1. EMAIL_TABLE = 'local_table'
1. Download [PostgresSQL]( https://www.postgresql.org/) and set up a local server that the application can connect to. 
1. The following command will take a while: Run ‘python run_data_processing.py’. This will preprocess all the emails in the maildir and run TextRank on batches of 1000 emails. The results will be stored in the local database. 
1. Once finished, you can now run the application locally. In a terminal, run ‘heroku local’. You will now be able to visit the website and explore the Enron emails. 

If you would like to have the data stored in a AWS RDS instance, just replace the ‘SQLALCHEMY_DATABASE_URI’ variable with the correct URL. For more information visit [here]( https://aws.amazon.com/getting-started/tutorials/create-connect-postgresql-db/)
Lastly to host this on Heroku, you can upload this repository to GitHub, then point Heroku to the repository and deploy. 

# Architecture Overview 
This web app was designed to be hosted on Heroku and store it's data in AWS RDS. 

![Architecture Overview](https://github.com/dailykirt/email_summary_prod/blob/master/Documents/Architecture%20Diagram.jpg)

# Production Flow 
![Production Overview](https://github.com/dailykirt/email_summary_prod/blob/master/Documents/Text%20Summarizing%20Flow%20Chart.jpg)
