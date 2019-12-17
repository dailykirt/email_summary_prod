#Local filepaths
mail_dir = '../data/maildir/'

#Used to look at single mailbox.
#mailbox = "phanis-s"
#mailbox = ["lenhart-m"]

#word embedding location
wb_file = '../data/glove.6B.300d.txt'

table = 'test_rank_db'

#local
POSTGRES_ADDRESS = 'localhost'
POSTGRES_USERNAME = 'postgres'
POSTGRES_PASSWORD = 'postgres'
POSTGRES_DBNAME = 'testdb'

#now create database connection string
#postgres_str = ('postgresql+psycopg2://{username}:{password}@{ipaddress}/{dbname}'
#                .format(username=POSTGRES_USERNAME,
#                        password=POSTGRES_PASSWORD,
#                        ipaddress=POSTGRES_ADDRESS,
#                        dbname=POSTGRES_DBNAME))

#AWS
postgres_str = 'postgresql+psycopg2://masterUsername:CraterLake12@rds-postgresql-email-sum.ced5yvd9vkk3.us-east-2.rds.amazonaws.com/myDatabase'
