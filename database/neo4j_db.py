from neo4j import GraphDatabase
import configparser
import os

class Neo4jGraphDB:
    def __init__(self, uri=None, user=None, password=None, database=None):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database

        # Read configuration file
        params = self._load_config(os.path.join(os.path.dirname(__file__), '../', 'config.ini'))
        uri = self.uri or params.get('uri', 'bolt://localhost:7687')
        user = self.user or params.get('user', 'neo4j')
        password = self.password or params.get('password', 'password')
        self._database = self.database or params.get('database', 'neo4j')

        # Create connection
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._session = None
    
    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config['neo4j']
    
    def _create_conn(self, uri, user, password):
        return GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()