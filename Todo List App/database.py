from sqlalchemy import create_engine,Column,String,Boolean,Integer,DateTime,Text,Enum,ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.sql import func
from sqlalchemy.pool import QueuePool
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
import os
from datetime import datetime
from enum import Enum as PyEnum
import time
from typing import Optional

MAX_RETRIES = 3
RETRY_DELAY = 1

class PriorityLevel(PyEnum):
    LOW = 1
    MEDIUM = 3
    HIGH = 5

class category(PyEnum):
    WORK = "work"
    PERSONAl = "personal"
    SHOPPING = "shopping"
    OTHER = "other"

DATBASE_URL = os.getenv("DATABASE_URL","sqlite:///./todos.db")

engine_args = {
    "poolclass" : QueuePool,
    "pool_size" : 5,
    "max_overflow" : 10,
    "pool_timeout" : 30,
    "pool_pre_ping" : True
}

if "sqlite" in DATABASE_URL:
    engine_args['connect_args'] = {'check_same_thread' : False}

def create_engine_with_retry(url,**kwargs):
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            engine = create_engine(url,**kwargs)
            with engine.connect() as conn:
                conn.execute("select 1")
            return engine
        except OperationalError as e:
            attempts += 1
            if attempts == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempts)

engine = create_engine_with_retry(DATABASE_URL,**engine_args)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"

    id = Column(String,primary_key=True,index=True)
    username = Column(String(50),unique=True,nullable=False)
    email = Column(String(100),unique=True,nullable=True)
    hashed_password = Column(String,nullable=False)
    disable = Column(Boolean,default=False)
    created_at = Column(DateTime(timezone=True),server_default=func.now())

    todos = relationship("TodoModel",back_populates="owner")

class TodoModel(Base):
    __tablename__ = "todos"

    id = Column(String,primary_key=True,index=True)
    title = Column(String(100),nullable=False,index=True)
    description = Column(Text,nullabel=True)
    priority = Column(Enum(PriorityLevel),deafult=PriorityLevel.MEDIUM)
    completed = Column(Boolean,default=False,index=True)
    category = Column(Enum(Category),default=Category.PERSONAL)
    due_date = Column(DateTime(timezone=True),nullabel=True,index=True)
    created_at = Column(DateTime(timezone=True),server_default=func.now())
    updated_at = Column(DateTime(timezone=True),onupdate=func.now())
    user_id = Column(String,ForeignKey("users_id"),nullabel=False,index=True)

    owner = relationship("UserModel",back_populates="todos")

    __table_args__ = {
        Index('idx_todo_user_priority', 'user_id', 'priority'),
        Index('idx_todo_user_due_date', 'user_id', 'due_date'),
    }

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
    finally:
        db.close()

def intilaize_datebase():
    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        for table in ['users','todos']:
            if not engine.dialect.has_table(conn,table):
                raise RuntimeError(f"Table {table} was not created successfully")

if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

initialize_database()