from __future__ import print_function, unicode_literals
from flask import Flask, flash, render_template
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flaskForms import SummarizationForm
import pandas as pd
from EmailModel import EmailModel
import pytz
import logging
import sys

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
utc=pytz.UTC

LOG_NAME = '__flask_app__'

def set_logger():
    """Configures the logger."""
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    format = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')
    handler.setFormatter(format)
    logger.addHandler(handler)
    return logger

@app.route('/', methods=['get','post'])
def home_page():
    logger = set_logger()
    model = EmailModel(db, Config.EMAIL_TABLE)
    form = SummarizationForm()
    employee_list = model.list_employees()
    form.inbox.choices = [(i, i) for i in employee_list ]

    if form.validate_on_submit():
        inbox = form.inbox.data
        timeframe = model.get_timeframe(inbox)
        start_date = form.start_date.data
        end_date = form.end_date.data

        if (start_date != '') and (end_date != ''):
            start = pd.to_datetime(start_date).replace(tzinfo=utc)
            end = pd.to_datetime(end_date).replace(tzinfo=utc)
            model.retrieve_summaries(start, end, inbox)

        if model.html_summary == [] or (form.show_dates.data == True) :
            flash('Please enter a start and end date between: ' + timeframe['start'] + ' and ' + timeframe['end'] + "<br/>")
        else:
            flash("Number of emails: " + str(len(model.enron_masked_df)) + "<br/>")
            if form.summary.data == True or form.summary_email.data == True:
                logger.info("Summary requested with the following parameters. Start: " + str(start) + " End: " + str(end) + " Inbox: " + inbox)
                flash("Number of summaries: " + str(len(model.html_summary)) + "<br/>")
                for count, summary in enumerate(model.html_summary):
                    flash(summary)
                    if form.summary_email.data == True:
                        flash("Full Email: " + model.original_emails[count] + "<br/>")
            else:
                logger.info("Full emails requested with the following parameters. Start: " + str(start) + " End: " + str(end) + " Inbox: " + inbox)
                model.display_emails()
                for email in model.html_emails:
                    flash(email)
            model.html_summary = []

    return render_template('home_page.html', form=form)

if __name__ == '__main__':
    app.run()