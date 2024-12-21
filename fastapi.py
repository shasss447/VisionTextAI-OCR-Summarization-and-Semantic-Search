from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import unquote
import logging

logger = logging.getLogger("uvicorn.error")  # FastAPI's default logger

# Initialize FastAPI app
app = FastAPI(title="Document Summary API")

# Database configuration
engine = create_engine("postgresql://postgres:12072024@localhost:5432/postgres")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Model
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, unique=True, index=True)
    content = Column(Text)
    summary=Column(Text)
# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/summary/{file_name}")
async def get_summary(file_name: str):
    """
    Get a summary of the document text for the given file name.
    """
    logger.info(f"Received file name: {file_name}")
    db = SessionLocal()
    try:
        # Query the document
        file_name = unquote(file_name).strip()
        logger.info(f"Normalized file name: {file_name}")
        document = db.query(Document).filter(Document.file_name == file_name).first()
        logger.info(f"Database query executed for file_name: {file_name}")
        if not document:
            logger.warning(f"No document found for file_name: {file_name}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"file_name": file_name, "summary": document.summary}
    
    except Exception as e:
        logger.error(f"Error during summary retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()