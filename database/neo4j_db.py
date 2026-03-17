from neo4j import GraphDatabase
import configparser
import os

class Neo4jGraphDB:

    def __init__(self, uri=None, user=None, password=None, database=None):
        params = self._load_config(os.path.join(os.path.dirname(__file__), '../', 'config.ini'))
        
        self._uri = uri or params.get('uri', 'bolt://localhost:7687')
        self._user = user or params.get('user', 'neo4j')
        self._password = password or params.get('password', 'password')
        self._database = database or params.get('database', 'neo4j')

        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
    
    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config['neo4j']

    def close(self):
        self._driver.close()