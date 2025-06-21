
import psycopg2
from abc import ABC, abstractmethod

from sqlalchemy import create_engine
from loguru import logger
import pandas as pd
from src.entity.config_entity import RDSConnectorConfig, RedshiftConnectorConfig

class DatabaseConnector(ABC):


    @abstractmethod
    def get_databases(self):
        """
        Retrieves a list of all databases available in the connected database server.

        Returns:
            tuple: A tuple containing the names of all databases.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_tables(self, dbname=''):
        """
        Retrieves a list of all tables available in the specified database.

        Args:
            dbname (str, optional): The name of the database to retrieve tables from. Defaults to an empty string, which means all databases will be considered.

        Returns:
            tuple: A tuple containing the names of all tables in the specified database.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_dataframe(self, query=""):
        """
        Retrieves a pandas DataFrame from the database using the provided query.

        Args:
            query (str, optional): The SQL query to retrieve the data. Defaults to an empty string.

        Returns:
            pandas.DataFrame: A DataFrame containing the data retrieved from the database.
        """
        raise NotImplementedError("Subclasses must implement this method.")



class RDSConnector(DatabaseConnector):

    def __init__(self, config: RDSConnectorConfig):
        self.config = config
        
    


    def get_databases(self):
        with create_engine(self.config.connection_string).connect() as conn:

            databases = conn.execute(
                                        "SHOW DATABASES"
                                        )

            databases = tuple([database[0] for database in databases])
            logger.info(f'No of Databases: {len(databases)}')

            return databases

        
            

    def get_tables(self, dbname=''):
        with create_engine(self.config.connection_string).connect() as conn:

            if dbname:
                conn.execute(
                                    f"USE {dbname}"
                                )
            else:
                raise ValueError("Gimme a DB name not an empty string :/")
                
            tables = conn.execute(
                                        "SHOW TABLES"
                                    )

            tables = tuple(table[0] for table in tables)
            logger.info(f'No of tables in {dbname}: {len(tables)}')

            return tables
        

    def get_dataframe(self, query=""):
        if query == "":
                raise ValueError("Forgot to pass a query bro?") 

        else:
            with create_engine(self.config.connection_string).connect() as conn:
            
                try:
                    df = pd.read_sql(
                            query,
                            con=conn
                            )
                    return df     
                except Exception as e:
                    raise e


class RedshiftConnector(DatabaseConnector):

    def __init__(self, config: RedshiftConnectorConfig):
        self.config = config
        

    def get_databases(self):
        
        query = "SELECT datname AS database_name FROM pg_database"
        df = self.get_dataframe(query,'analytics')
        
        databases = tuple(df.database_name.values)
        logger.info(f'No of Databases: {len(databases)}')

        return databases

        
    def get_tables(self, dbname=''):
        dbname, schema = dbname.split('.')

        query = f"""SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = '{schema}'"""
        df = self.get_dataframe(query, dbname)
        
        tables = tuple(df.tablename.values)

        logger.info(f'No of tables in {dbname}: {len(tables)}')

        return tables
        

    def get_dataframe(self, query="", dbname=""):      
        if query == "" or dbname == "":
            raise ValueError("Please check if the query or dbname is provided..") 

        else:
            try:
                connection_string = f"""dbname={dbname} 
                                        host={self.config.host} 
                                        password={self.config.password} 
                                        user={self.config.username} 
                                        port={self.config.port}"""
                
                with psycopg2.connect(connection_string) as conn:
                    df = pd.read_sql(
                            query,
                            con=conn
                            )
                    return df
            except Exception as e:
                raise e

    

