#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Ei Tsujimura
#
# Created:     30/04/2025
# Copyright:   (c) Ei Tsujimura 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from sqlmodel import SQLModel, Field, Relationship, Session, create_engine, select
from pydantic import BaseModel, constr  # added by HTMX proj
from typing import List, Optional, Annotated
import datetime


################################################################################
# vehicle data model
################################################################################

class VehicleBase(SQLModel):
    vin: str = Field(index=True, primary_key=True)
    model: str  = ""
    date: datetime.date | None = None
    odometer: int | None = None
    status: str = ""


class Vehicle(VehicleBase, table=True):
    __table_args__ = {'extend_existing': True}  # for making same_name table
##    id: int | None = Field(default=None)


class VehiclePublic(VehicleBase):
    vin: str


class VehicleCreate(VehicleBase):
    secret_name: str   # just an example


class VehicleUpdate(VehicleBase):
    vin: str = Field(index=True)
    date: datetime.date | None = None
    odometer: int | None = None
    status: str | None = None
    secret_name: str | None = None


################################################################################
# ticket data model
################################################################################
class TicketBase(SQLModel):
    vin: str = Field(index=True, foreign_key="vehicle.vin")
    date_init: datetime.date | None = None
    date_update: datetime.date | None = None
    status: str  = ""
    labour: float | None = None


class Ticket(TicketBase, table=True):
    __table_args__ = {'extend_existing': True}  # for making same_name table
    id: int | None = Field(default=None, primary_key=True)
    tags: str | None = ""  # None (NULL acceptable)


################################################################################
# parts data model
################################################################################
class Parts(SQLModel, table=True):
    # let parts list handle description , etc
    id: int | None = Field(default=None, primary_key=True)
    ticket_id : int = Field(foreign_key="ticket.id")
    ref : str | None = None  # Non-Foreign key for unknown parts #
    cost : float | None = None
    order_id : int | None = None
    eta : datetime.date | None = None
    ata : datetime.date | None = None  # actual time delivered
    tags : str | None = "" # None (NULL acceptable)


################################################################################
# HTMX test
################################################################################
class NoteItem(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    item: str

class NoteItemCreate(BaseModel):
    item: constr(min_length=2, max_length=100)  # Validate item length


################################################################################
# User auth
################################################################################
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int = Field(default=None, primary_key=True)
    username : str
    hashed_password : str
    disabled : bool = True
    role : str | None = None
    print_name : str | None = None
    email : str | None = None

