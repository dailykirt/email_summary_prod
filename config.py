import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'my very secret key'
    #local
    #SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/testdb'
    #AWS
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://masterUsername:CraterLake12@rds-postgresql-email-sum.ced5yvd9vkk3.us-east-2.rds.amazonaws.com/myDatabase'
    #Enviroment URL
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://masterUsername:CraterLake12@rds-postgresql-email-sum.ced5yvd9vkk3.us-east-2.rds.amazonaws.com/myDatabase?sslrootcert=rds-combined-ca-bundle.pem&sslmode=require'