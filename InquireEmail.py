from __future__ import print_function, unicode_literals
from PyInquirer import prompt
from EmailModel import EmailModel

"""This file provides an interactive API to interact with the Enron dataset"""
class InquireEmail:
    def __init__(self):
        # Load model data
        self.model = EmailModel()
        self.employee_list = self.model.list_employees()

    def ask_inbox(self):
        inbox_question = [
            {
                'type': 'list',
                'name': 'inbox',
                'message': 'Whose inbox would you like to summarize?',
                'choices':  self.employee_list
            },
        ]
        self.inbox_answer = prompt(inbox_question)

    def get_date_range(self):
        dates = self.model.get_timeframe(self.inbox_answer['inbox'])
        self.date_range = dates['start'] + " and " + dates['end']

    def ask_dates(self):
        self.get_date_range()
        date_question = [
            {
            'type': 'input',
            'name': 'start',
            'message': "Please input a start date between " + self.date_range,
            },
            {
            'type': 'input',
            'name': 'end',
            'message': "Please input a end date between " + self.date_range,
            },
        ]
        self.date_answer = prompt(date_question)


    def get_summary(self):
        self.model.summarize_emails(self.date_answer['start'], self.date_answer['end'], self.inbox_answer['inbox'])

    def start_ask(self):
        self.ask_inbox()
        self.ask_dates()
        self.get_summary()