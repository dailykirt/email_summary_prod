from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, StringField, RadioField, validators


class SummarizationForm(FlaskForm):
    #model = EmailModel()
    #employee_list = employeeChoiceList(model.list_employees())
    #inbox = SelectField(u'Inbox', choices= employeeChoiceList(employee_list))
    inbox = SelectField('Inbox')
    submit = SubmitField('Submit')
    start_date = StringField('Start Date: ')
    end_date = StringField('End Date: ')

    display_full = RadioField(
        'display_full',
        [validators.Required()],
        choices=[('yes', True), ('no', False)], default='no'
    )

