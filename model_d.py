
from sqlalchemy import Column, Integer, JSON, String, create_engine
from database import Base

class Face(Base):
    __tablename__ = 'face_data_2'
    
    id = Column(String, primary_key=True, index=True)
    feature = Column(JSON)
    # features = Column(JSONB, nullable=False)
