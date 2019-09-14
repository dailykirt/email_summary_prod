from __future__ import print_function, unicode_literals
from flask import Flask, flash, render_template
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flaskForms import SummarizationForm
import pandas as pd
from EmailModel import EmailModel
import pytz

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
#Instantiate model with conneciton to database.
model = EmailModel(db)
utc=pytz.UTC

@app.route('/', methods=['get','post'])
def home_page():
    form = SummarizationForm()
    employee_list = model.list_employees()
    form.inbox.choices = [(i, i) for i in employee_list ]
    start_date = ''
    end_date = ''

    if form.validate_on_submit():
        inbox = form.inbox.data
        timeframe = model.get_timeframe(inbox)
        start_date = form.start_date.data
        end_date = form.end_date.data
        display_full = form.display_full.data

        if (start_date != '') and (end_date != ''):
            start = pd.to_datetime(start_date).replace(tzinfo=utc)
            end = pd.to_datetime(end_date).replace(tzinfo=utc)
            model.summarize_emails(start, end, inbox)
        if model.final_summary == '':
            flash('Please enter a start and end date between: ' + timeframe['start'] + ' and ' + timeframe['end'])
        else:
            for count, summary in enumerate(model.html_summary):
                flash(summary)
                if display_full == 'yes':
                    flash("Full Email: " + model.original_emails[count] + "<br/>")

    return render_template('home_page.html', form=form)

if __name__ == '__main__':
    app.run()
