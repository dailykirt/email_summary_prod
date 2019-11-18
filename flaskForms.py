from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, StringField, RadioField, validators


class SummarizationForm(FlaskForm):
    inbox = SelectField('Inbox')
    summary = SubmitField('Summaries')
    email = SubmitField('Emails')
    summary_email = SubmitField('Summaries and emails')
    show_dates = SubmitField('Display valid dates')
    start_date = StringField('Start Date: ')
    end_date = StringField('End Date: ')
