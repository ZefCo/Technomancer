from contextvars import ContextVar

current_user: ContextVar[str] = ContextVar("current_user", default = "Server")

def set_current_user(username: str):
    current_user.set(username or "Anonymous")

def get_current_user() -> str:
    return current_user.get()