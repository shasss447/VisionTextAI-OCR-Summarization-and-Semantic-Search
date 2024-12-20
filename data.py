import psycopg2
from psycopg2.extras import execute_values
from pymilvus import (MilvusClient,DataType,AnnSearchRequest,RRFRanker)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
from PIL import Image
from typing import Dict, List, Any
import logging
import torch
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class PostGreDatabaseHandler:
    def __init__(self, postgres_conn_string: str):
        """
        Initialize database connections and models.
        
        Args:
            postgres_conn_string: PostgreSQL connection string

        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize PostgreSQL connection
        self.pg_conn = psycopg2.connect(postgres_conn_string)

        """
        Create the necessary tables in PostgreSQL database.
    
        Args:
        conn_params (dict): Database connection parameters
        """
        commands = (
        """
        CREATE TABLE IF NOT EXISTS documents (
           id SERIAL PRIMARY KEY,
           file_name VARCHAR(255) NOT NULL,
           content TEXT NOT NULL,
           summary TEXT NOT NULL,
           confidence FLOAT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS named_entities (
           id SERIAL PRIMARY KEY,
           doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
           entity TEXT NOT NULL,
           label VARCHAR(50) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS pos_tags (
           id SERIAL PRIMARY KEY,
           doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
           token TEXT NOT NULL,
           pos_tag VARCHAR(20) NOT NULL
        )
        """
        )
    
        try:
            with self.pg_conn.cursor() as cur:

              # Create each table
              for command in commands:
                cur.execute(command)
            
              cur.close()
              self.pg_conn.commit()
        
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")

    def store_document_data(self, doc_data: List[Dict[str, Any]]):
      """
      Store document data including file name, text, entities, and POS tags into PostgreSQL tables.
    
      Args:
        doc_data (List[Dict[str, Any]]): Dictionary containing:
            - file_name (str): Name of the file
            - text (str): Content of the file
            - summary (str): Summary of the content of the file
            - confidence_score (float): Confidence score for the document
            - entities (Dict[str, str]): Dictionary of entity:label pairs
            - pos (Dict[str, str]): Dictionary of text:pos_tag pairs
            - postgres_conn_string: PostgreSQL connection string
      """
      try:
        with self.pg_conn.cursor() as cur:
            for data in doc_data:
                # First, insert the document and get its ID
                doc_insert_query = """
                  INSERT INTO documents (file_name, content,summary, confidence)
                  VALUES (%s, %s, %s, %s)
                  RETURNING id;
                """
                cur.execute(doc_insert_query, (data['file_name'],data['processed_text'],data['summary'],data['confidence_scores']))
                doc_id = cur.fetchone()[0]
            
                # Insert named entities
                if data.get('entities'):
                  entities_insert_query = """
                    INSERT INTO named_entities (doc_id, entity, label)
                    VALUES (%s, %s, %s);
                """
                  entity_data = [
                    (doc_id, entity, label)for entity, label in data['entities']
                    ]
                cur.executemany(entities_insert_query, entity_data)
            
                # Insert POS tags
                if data.get('pos_tags'):
                  pos_insert_query = """
                    INSERT INTO pos_tags (doc_id, token, pos_tag)
                    VALUES (%s, %s, %s);
                """
                  pos_data = [
                    (doc_id, token, pos_tag)for token, pos_tag in data['pos_tags']
                    ]
                cur.executemany(pos_insert_query, pos_data)
            
            self.pg_conn.commit()
            
      except Exception as e:
        self.pg_conn.rollback()
        raise Exception(f"Error storing document data: {str(e)}")
      
    def postgres_file_entities(self, file_name: str) -> dict:
        """
        Search for a specific file name and return entities from the documents table.

        Args:
         file_name (str): Exact file name to search for

        Returns:
         dict: including entities and label.
              Returns empty dict if file not found.
        """
        query = """
          SELECT entity,label
          FROM named_entities
          JOIN documents ON named_entities.doc_id = documents.id
          WHERE documents.file_name = %s;
        """
    
        try:
          cur = self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
          cur.execute(query, (file_name,))
          result = cur.fetchall()
          self.pg_conn.commit()
          return result if result else {}
          
        except Exception as e:
          self.logger.error(f"Error in file search: {str(e)}")
          if self.pg_conn:
            self.pg_conn.rollback()  # Rollback the transaction on error
        return {}
    
    def postgres_entities_file(self, entity: str) -> dict:
        """
        Search for entities and return file names.

        Args:
         entity (str): entity to search for

        Returns:
         dict: including file names.
              Returns empty dict if file not found.
        """
        query = """
          SELECT distinct file_name
          FROM documents
          JOIN named_entities ON documents.id=named_entities.doc_id
          WHERE named_entities.label = %s;
        """
    
        try:
          cur = self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
          cur.execute(query, (entity,))
          result = cur.fetchall()
          self.pg_conn.commit()
          return result if result else {}
        
        except Exception as e:
          self.logger.error(f"Error in file search: {str(e)}")
          if self.pg_conn:
            self.pg_conn.rollback()  # Rollback the transaction on error
        return {}
      
    def postgres_file_summary(self, file_name: str) -> dict:
        """
        Search for a specific file name and return its details from the documents table.

        Args:
         file_name (str): Exact file name to search for

        Returns:
         dict: Document details including file name, content, and confidence score.
              Returns empty dict if file not found.
        """
        query = """
          SELECT
          summary
          FROM documents
          WHERE file_name = %s
          LIMIT 1;
        """
    
        try:
          cur = self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
          cur.execute(query, (file_name,))
          result = cur.fetchone()
          self.pg_conn.commit()  # Commit the transaction
          return result if result else {}
        
        except Exception as e:
          self.logger.error(f"Error in file search: {str(e)}")
          if self.pg_conn:
            self.pg_conn.rollback()  # Rollback the transaction on error
        return {}
    
    def postgrekeyword_search(self, keyword: str) -> List[dict]:
       """
       Search for documents containing the given keyword and return their details.
    
       Args:
        keyword (str): Keyword to search for in documents content and file names
    
      Returns:
        List[dict]: List of matching documents with their file names and content
       """
       query = """
        SELECT DISTINCT 
        file_name, content
        FROM documents
        WHERE content ILIKE %s OR file_name ILIKE %s
        ORDER BY file_name;
        """
       try:
         cur = self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
         # Add wildcards for partial matching
         search_pattern = f'%{keyword}%'
         cur.execute(query, (search_pattern, search_pattern))
         results = cur.fetchall()
         return results
            
       except Exception as e:
         self.logger.error(f"Error in keyword search: {str(e)}")
       if self.pg_conn:
         self.pg_conn.rollback()  # Rollback the transaction on error
       return []

    def close(self):
      """Close database connections."""
      if hasattr(self, 'pg_conn') and not self.pg_conn.closed:
            self.pg_conn.close()
      """Close database connections."""
      if hasattr(self, 'pg_conn') and not self.pg_conn.closed:
            self.pg_conn.close()

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

class MilvusDatabaseHandler:
    def __init__(self,milvusdb:str):
        """
        Initialize database connections and models.

        Args:
            postgres_conn_string: PostgreSQL connection string

        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize Milvus Client
        self.client=MilvusClient(milvusdb)

        # Embedding Model
        self.bge=BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3',
        device='cpu',
        use_fp16=False
        )

        # Image Embedding
        self.extractor = FeatureExtractor("resnet34")

        # Creating schema for texts
        text_schema=MilvusClient.create_schema(
            enable_dynamic_filter=True,
            )

        # Addings fields
        text_schema.add_field(field_name="file_name",datatype=DataType.VARCHAR,is_primary=True,max_length=100)
        text_schema.add_field(field_name="Text",datatype=DataType.VARCHAR,max_length=1000)
        text_schema.add_field(field_name="Summary",datatype=DataType.VARCHAR,max_length=100)
        text_schema.add_field(field_name="sparse",datatype=DataType.SPARSE_FLOAT_VECTOR)
        text_schema.add_field(field_name="dense",datatype=DataType.FLOAT_VECTOR,dim=self.bge.dim["dense"])
        text_schema.add_field(field_name="confidence",datatype=DataType.FLOAT)

        # Creating Index
        text_index_param=self.client.prepare_index_params()
        text_index_param.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="AUTOINDEX",
            metric_type="IP",
            params={"nlist":128}
            )

        text_index_param.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build":0.2},
            )

        # Creating Collection
        if self.client.has_collection(collection_name="Text_collection"):
            self.client.drop_collection(collection_name="Text_collection")
        self.client.create_collection(
             collection_name="Text_collection",
             schema=text_schema,
             index_params=text_index_param
            )

        # Creating schema for Named-Entities
        named_schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_filter=True,
            )

        # Adding fields
        named_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        named_schema.add_field(field_name="file_name", datatype=DataType.VARCHAR,max_length=1000)
        named_schema.add_field(field_name="entity", datatype=DataType.VARCHAR, max_length=200)
        named_schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=50)

        # Creating Collection
        if self.client.has_collection(collection_name="Named_collection"):
            self.client.drop_collection(collection_name="Named_collection")
        self.client.create_collection(
             collection_name="Named_collection",
             schema=named_schema,
            )

        # Creating schema for Pos
        pos_schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            )

        # Adding fields
        pos_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        pos_schema.add_field(field_name="file_name", datatype=DataType.VARCHAR,max_length=1000)
        pos_schema.add_field(field_name="token", datatype=DataType.VARCHAR, max_length=100)
        pos_schema.add_field(field_name="pos", datatype=DataType.VARCHAR, max_length=20)

        # Creating Collection
        if self.client.has_collection(collection_name="Pos_collection"):
            self.client.drop_collection(collection_name="Pos_collection")
        self.client.create_collection(
             collection_name="Pos_collection",
             schema=pos_schema,
            )

        # Creating schema for image
        img_schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            )

        # Adding fields
        img_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        img_schema.add_field(field_name="file_name", datatype=DataType.VARCHAR,max_length=1000)
        img_schema.add_field(field_name="img_file_name", datatype=DataType.VARCHAR, max_length=100)
        img_schema.add_field(field_name="img_file_path", datatype=DataType.VARCHAR, max_length=100)
        img_schema.add_field(field_name="dense",datatype=DataType.FLOAT_VECTOR,dim=512)

        # Creating Index
        img_index_param=self.client.prepare_index_params()
        img_index_param.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={"nlist":128}
            )

        # Creating collection
        if self.client.has_collection(collection_name="Img_collection"):
            self.client.drop_collection(collection_name="Img_collection")
        self.client.create_collection(
             collection_name="Img_collection",
             schema=img_schema,
             index_params=img_index_param
            )

    def store_document_data(self,doc_data:List[Dict[str, Any]]):
      """
      Store document data including file name, text, entities, and POS tags into Milvus Collections.

      Args:
        doc_data (List[Dict[str, Any]]): Dictionary containing:
            - file_name (str): Name of the file
            - text (str): Content of the file
            - summary (str): Summary of the content of the file
            - confidence_score (float): Confidence score for the document
            - entities (Dict[str, str]): Dictionary of entity:label pairs
            - pos_tags (Dict[str, str]): Dictionary of text:pos_tag pairs
            - images (list(Dict[str,str])): List of dictionary of images
      """

      try:
        for data in doc_data:
            # Generate text embeddings
            embeddings = self.bge.encode_documents([data['processed_text']])
            sparse_vector = embeddings['sparse']
            dense_vector = embeddings['dense'][0]
            # Insert into Text_collection
            text_data = [{
                'file_name': data['file_name'],
                'Text': data['processed_text'],
                'summary':data['summary'],
                'sparse': sparse_vector,
                'dense': dense_vector,
                'confidence': data['confidence_scores']
            }]

            text_insert_result = self.client.insert(
                collection_name="Text_collection",
                data=text_data
            )

            # Insert into Named_collection
            if data.get('entities'):
                named_data = [
                    {
                        'file_name': data['file_name'],
                        'entity': entity,
                        'label': label
                    }
                    for entity, label in data['entities']
                ]
                self.client.insert(
                    collection_name="Named_collection",
                    data=named_data
                )

            # Insert into Pos_collection
            if data.get('pos_tags'):
                pos_data = [
                    {
                        'file_name': data['file_name'],
                        'token': token,
                        'pos': pos
                    }
                    for token, pos in data['pos_tags']
                ]
                self.client.insert(
                    collection_name="Pos_collection",
                    data=pos_data
                )

            # Insert into Img_collection
            if data.get('extracted_images'):
                for img_info in data['extracted_images']:
                    img_embedding = self.extractor(img_info['path'])
                    img_data = [{
                        'file_name': data['file_name'],
                        'img_file_name': img_info['filename'],
                        'img_file_path': img_info['path'],
                        'dense': img_embedding.tolist()
                    }]
                    self.client.insert(
                        collection_name="Img_collection",
                        data=img_data
                    )

            # Flush collections to ensure data is written
            self.client.flush(collection_name="Text_collection")
            self.client.flush(collection_name="Named_collection")
            self.client.flush(collection_name="Pos_collection")
            self.client.flush(collection_name="Img_collection")

      except Exception as e:
            raise Exception(f"Error inserting data into Milvus: {str(e)}")

    def milvuskeyword_search(self,keyword:str)->List[Dict[str, Any]]:
      """
        Perform hybrid search using both sparse and dense vectors.

        Args:
            keyword (str): Search keyword/query

        Returns:
            List[Dict]: List of search results with scores and metadata
      """

      try:

        # Generate text embeddings
        embeddings = self.bge.encode_documents([keyword])
        sparse_vector = embeddings['sparse']
        dense_vector = embeddings['dense'][0]

        search_par_1={
            "data":[dense_vector],
            "anns_field":"dense",
            "param":{
                "metric_type":"IP",
                "params":{"nprobe":10}
                },
                "limit":2
                }
        req1=AnnSearchRequest(**search_par_1)

        search_par_2={
            "data":[sparse_vector],
            "anns_field":"sparse",
            "param":{
                "metric_type":"IP",
                "params":{"drop_ratio_build":0.2}
                },
                "limit":2
                }
        req2=AnnSearchRequest(**search_par_2)

        reqs=[req1,req2]
        rank=RRFRanker(100)

        res = self.client.hybrid_search(
            collection_name="Text_collection",
            reqs=reqs,
            ranker=rank,
            limit=4
            )
        result=[]
        for hits in res:
         for hit in hits:
          d={
              'file_name':hit['id'],
              'score':hit['distance'],
              'text':self.client.get(
                  collection_name="Text_collection",
                  ids=[hit['id']],
                  output_fields=["Text"]
              )
          }
          result.append(d)
        return result

      except Exception as e:
        raise Exception(f"Error performing hybrid search: {str(e)}")

    def milvusimage_search(self,path:str)->List[Dict]:
        """
        Search for similar images in the database.

        Args:
            image_path (str): Path to the query image
            limit (int): Maximum number of results to return
            distance_threshold (float): Maximum distance threshold for similarity

        Returns:
            List[Dict]: List of similar images with metadata
        """
        try:
             embeddings=self.extractor(path)

             search_params = {
                "metric_type": "COSINE",
            }

             # Execute search
             results = self.client.search(
                "Img_collection",
                data=[embeddings],
                anns_field="dense",
                search_params=search_params,
                limit=2,
                output_fields=["file_name", "img_file_name", "img_file_path"]
            )
             # Process results
             processed_results = []
             for hits in results:
                for hit in hits:
                    result = {
                        "distance":hit['distance'],
                        "image_file_name": hit['entity']['img_file_name'],
                        "image_file_path": hit['entity']['img_file_path'],
                        "document_name": hit['entity']['file_name'],
                        "document_text": self.client.get(
                                         collection_name="Text_collection",
                                         ids=[hit['entity']['file_name']],
                                         output_fields=["Text"]
              )
                    }
                    processed_results.append(result)

             return processed_results

        except Exception as e:
            raise Exception(f"Error performing image search: {str(e)}")

    def close(self):
        """Release collection resources."""
        self.text_collection.release()
        self.named_collection.release()
        self.pos_collection.release()
        self.img_collection.release()


        """Release collection resources."""
        self.text_collection.release()
        self.named_collection.release()
        self.pos_collection.release()
        self.img_collection.release()


        """Release collection resources."""
        self.text_collection.release()
        self.named_collection.release()
        self.pos_collection.release()
        self.img_collection.release()
