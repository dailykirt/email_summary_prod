from __future__ import print_function, unicode_literals
from flask import Flask, flash, render_template
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flaskForms import SummarizationForm
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

@app.route('/', methods=['get','post'])
def home_page():
    form = SummarizationForm()

    if form.validate_on_submit():
        inbox = form.inbox.data
        timeframe = form.model.get_timeframe(inbox)
        start_date = form.start_date.data
        end_date = form.end_date.data
        display_full = form.display_full.data
        test_df = pd.read_sql_query('select * from enron_single_mailbox', con=db.engine)
        print(test_df.head())

        if (start_date != '') and (end_date != ''):
            form.model.summarize_emails(start_date,end_date, inbox)
        if form.model.final_summary == '':
            flash('Please enter a start and end date between: ' + timeframe['start'] + ' and ' + timeframe['end'])
        else:
            for count, summary in enumerate(form.model.html_summary):
                flash(summary)
                if display_full == 'yes':
                    flash("Full Email: " + form.model.original_emails[count] + "<br/>")

    return render_template('home_page.html', form=form)

if __name__ == '__main__':
    app.run()
    #api_start = InquireEmail()
    #api_start.start_ask()

