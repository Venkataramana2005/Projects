from fastapi import FastAPI,HTTPException,Depends,status,Query
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from pydantic import BaseModel,Field,validator
from typing import List,Optional
import uvicorn
from datetime import datetime,timedelta
import uuid
from enum import Enum

app = FastAPI(
    title="Advanced Todo API",
    description="A compherensive todo application with authenitication, database integration, and advanced features",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class PriorityLevel(int,Enum):
    Low = 1
    MEDIUM = 3
    HIGH = 4

class Category(str,Enum):
    WORK = "work"
    PERSONAL = "personal"
    SHOPPING = "shopping"
    OTHER = "other"

class TodoBase(BaseModel):
    title: str = Field(...,min_length=1,max_length=100,description="Todo title")
    description: Optional[str] = Field(None,max_length=500,description="Todo description")
    priority: PriorityLevel = Field(PriorityLevel.MEDIUM,description="Priority level")
    completed: Category = Field(Category.PERSONAL,description="Optional due date")
    due_date: Optional[datetime] = Field(None,description="Optional due date")

    @validator('due_date')
    def due_date_must_be_future(cls,v):
        if v and v < datetime.utcnow():
            raise ValueError("Due date must be in the future")
        return v

class TodoCreate(TodoBase):
    pass

class TodoUpdate(BaseModel):
    title: Optional[str] = Field(None,min_length=1,max_length=100)
    description: Optional[str] = Field(None,max_length=500)
    priority: Optional[PriorityLevel] = None
    completed: Optional[bool] = None
    category: Optional[Category] = None
    due_date: Optional[datetime] = None

    @validator('due_date')
    def due_date_must_be_future(cls,v):
        if v and v < datetime.utcnow():
            raise ValueError("Due date must be in the future")
        return v

class Todo(TodoBase):
    id: str = Field(...,description="Unique todo identifier")
    created_at: datetime = Field(...,description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None,description="Last update timestamp")
    owner_id: str = Field(...,description="User ID who owns this todo")

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class User(BaseModel):
    id: str
    username: str
    disabled: bool = False

todos_db = {}
users_db = {
    "testuser" : {
        "id" : "user1",
        "username" : "testuser",
        "password" : "testpassword",
        "disabled" : False
    }
}

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = users_db.get(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invallid authentication credentials",
            headers = {"WWW-Authentication":"Bearer"}
        )
    return user

@app.post("/todos/",response_model=Todo,status_code=status.HTTP_201_CREATED)
async def create_todo(
    todo: TodoCreate,
    current_user: User = Depends(get_current_user)
):
    todo_id = str(uuid.uuid4())
    now = datetime.utcnow()

    new_todo = Todo(
        id=todo_id,
        title=todo.title,
        description=todo.description,
        priority=todo.priority,
        completed=todo.completed,
        due_date=todo.due_date,
        created_at=now,
        updated_at=now,
        owner_id=current_user["id"]
    )

    todos_db[todo_id] = new_todo
    return new_todo

@app.get("/todos/",response_model=List[Todo])
async def get_todos(
    current_user: User = Depends(get_current_user),
    skip: int = Query(0,ge=0),
    limit: int = Query(100,ge=1,le=1000),
    completed: Optional[bool] = None,
    priority: Optional[PriorityLevel] = None,
    category: Optional[Category] = None,
    overdue: Optional[bool] = None
):
    user_todos = [todo for todo in user_todos if todo.owner_id == current_user["id"]]

    if completed is not None:
        user_todos = [todo for todo in user_todos if todo.completed == completed]
    if priority is not None:
        user_todos = [todo for todo in user_todos if todo.priority == priority]
    if category is not None:
        user_todos = [todo for todo in user_todos if todo.category == category]
    if overdue is not None:
        now = datetime.utcnow()
        user_todos = [todo for todo in user_todos
                        if todo.due_date and (todo.due_date < now) == overdue]
    return user_todos[skip:skip+limit]

@app.get("/todos/{todo_id}",response_model=List[Todo])
async def get_todo(
    todo_id: str,
    current_user: User = Depends(get_current_user)
):
    if todo_id not in todos_db or todos_db[todo_id].owner_id != current_user["id"]:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Todo not found or access denied"
        )
    return todos_db[todo_id]

@app.put("/todos/{todo_id}",response_model=Todo)
async def update_todo(
    todo_id: str,
    todo_update: TodoUpdate,
    current_user: User = Depends(get_current_user)
):
    if todo_id not in todos_db or todos_db[todo_id].owner_id != current_user["id"]:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Todo not found or access denied"
        )
    
    existing_todo = todos_db[todo_id]
    update_todo = todo_update.dict(exclude_unset=True)

    for field,value in update_data.items():
        setattr(existing_todo,field,value)

    existing_todo.updated_at = datetime.utcnow()
    todos_db[todo_id] = existing_todo
    
    return existing_todo

@app.delete("/todos/{todo_id}",status_code=status.HTTP_204_NO_CONTENT)
async def delete_todo(
    todo_id: str,
    current_user: User = Depends(get_current_user)
):
    if todo_id not in todos_db or todos_db[todo_id].owner_id != current_user['id']:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Todo not found or access denied"
        )
    
    del todos_db[todo_id]

@app.get("/todos/stats/summary")
async def get_todo_stats(current_user: User = Depends(get_current_user)):
    user_todos = [todo for todo in todos_db.values() if todo.owner_id == current_user["id"]]
    total = len(user_todos)
    completed = sum(1 for todo in user_todos if todo.completed)
    pending = total - completed
    
    priority_counts = {level.name: 0 for level in PriorityLevel}
    for todo in user_todos:
        priority_counts[todo.priority.name] += 1
 
    category_counts = {cat.value: 0 for cat in Category}
    for todo in user_todos:
        category_counts[todo.category.value] += 1
    
    now = datetime.utcnow()
    overdue = sum(1 for todo in user_todos 
                 if todo.due_date and not todo.completed and todo.due_date < now)
    
    return {
        "total": total,
        "completed": completed,
        "pending": pending,
        "overdue": overdue,
        "completion_rate": round(completed / total * 100, 2) if total > 0 else 0,
        "priority_distribution": priority_counts,
        "category_distribution": category_counts
    }

@app.get("/todos/upcoming/", response_model=List[Todo])
async def get_upcoming_todos(
    current_user: User = Depends(get_current_user),
    days: int = Query(7, description="Number of days to look ahead")
):
    now = datetime.utcnow()
    end_date = now + timedelta(days=days)
    
    upcoming = [
        todo for todo in todos_db.values()
        if todo.owner_id == current_user["id"]
        and todo.due_date
        and now <= todo.due_date <= end_date
        and not todo.completed
    ]

    upcoming.sort(key=lambda x: x.due_date)
    return upcoming

if __name__ == "__main__":
    uvicorn.run("app:app",reload=True)