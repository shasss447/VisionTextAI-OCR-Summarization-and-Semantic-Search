import os

from data import PostGreDatabaseHandler,MilvusDatabaseHandler
from extracter import TesseractProcessor

import os

def get_image_names(directory_path):
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_names = []
    
    # Get all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(image_extensions):
            image_names.append(filename)
            
    return image_names


Tprocessor = TesseractProcessor()

#change directory name to add images in the database
news_directory = "dataset/news"
filenames = get_image_names(news_directory)

Tresult=[]
for filename in filenames:
    Tresult.append(Tprocessor.process_document(filename))

pgdb=PostGreDatabaseHandler("postgresql://db:password@localhost:5432/postgres")
pgdb.store_document_data(Tresult)

mvdb=MilvusDatabaseHandler("/milvus_demo.db")
mvdb.store_document_data(Tresult)