#from typing import Any, Dict, Optional
from pydantic import BaseSettings #, PostgresDsn, validator
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    PORT: int
    DOMAIN: str

    class Config:
        _env_file=".env"

settings = Settings()