# Extractive Summarization Application
This is the code that can be hosted to explore the Enron email dataset using extractive summarization methods. 

# Web Interface
You can explore the Enron summaries [here](http://enron-emails.herokuapp.com/). 
Select an inbox then click on 'Display valid dates' to see the time range of emails you can summarize. Then click on one of the following buttons:

'Summaries': Produces extractive summaries for all the emails in a given time frame. 
'Emails': Displays every email. 
'Summaries and emails': Give summaries, but also show the email the extractive summary was found in. 

# Architecture Overview 
This web app was designed to be hosted on Heroku and store it's data in AWS RDS. 

![Architecture Overview](https://github.com/dailykirt/email_summary_prod/blob/master/Documents/Architecture%20Diagram.jpg)

# Production Flow 
![Production Overview](https://github.com/dailykirt/email_summary_prod/blob/master/Documents/Text%20Summarizing%20Flow%20Chart.jpg)
