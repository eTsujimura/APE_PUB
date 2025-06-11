#-------------------------------------------------------------------------------
# Name:        db_init
# Purpose:     This is not nessesary so far, only for creating db separately
#
# Author:      U640919
#
# Created:     30/04/2025
# Copyright:   (c) U640919 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# database.py
from sqlmodel import SQLModel, create_engine
import models

DATABASE_URL = "sqlite:///./ape.sqlite"  # avoid overwriting ape.sqlite
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

if __name__ == "__main__":
    create_db_and_tables()