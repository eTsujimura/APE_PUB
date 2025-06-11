#-------------------------------------------------------------------------------
# Name:        APE
# Purpose:
#
# Author:      Ei Tsujimura
#
# Created:     28/03/2025
# Copyright:   (c) Ei Tsujimura 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------


from fastapi import FastAPI, Response, Request, Depends, HTTPException, Query, Form, UploadFile, status, APIRouter, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# for user auth
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from jwt import JWTError

# middleware for user auth
from starlette.middleware.base import BaseHTTPMiddleware



# datetime for user auth, etc
from datetime import datetime, timedelta, timezone

# for import file
import csv
import shutil
import pprint
from tempfile import NamedTemporaryFile
from pathlib import Path

# for SQL model, can I leave it to models.py?
##from sqlmodel import Field, Session, SQLModel, create_engine, select
##from typing import Annotated, List

# for SQL models use * for easy update on models.py
from models import *

import sys

# for data handling
import pandas as pd
import csv
from io import StringIO, BytesIO

# for report
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, portrait


################################################################################
# Environmental variables
################################################################################

# Acess token SSL
SECRET_KEY = "f6079d1490cf1675ea4bb309c650689b6712fb20b9a98b30d91fd5331274b18e"
ALGORITHM = "HS256"

# counting units
KB = 1024
MB = 1024 * KB



################################################################################
# DB engine, session & dependency
# engine : database
# session : connection to db
################################################################################

sqlite_file_name = "ape.sqlite"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}

# echo true for debugging only
engine = create_engine(sqlite_url, connect_args=connect_args, echo=False)


def create_db_and_tables():
    """
    get table from metadata (existing data model classes) and create
    """
    SQLModel.metadata.create_all(engine)
    db_tables = SQLModel.metadata.tables
    sys.stderr.write("{}\n".format(db_tables.keys()))


# A Session is what stores the objects in memory
# and keeps track of any changes needed in the data,
# then it uses the engine to communicate with the database.
# yield provides new Session for each request.
def get_session():
    with Session(engine) as session:
        yield session


# create session dependency
# not clear to me --> run get_session and recieve the return(Session class)
SessionDep = Annotated[Session, Depends(get_session)]


################################################################################
# User auth preparation
# Some func for dependency
# Auth middleware
################################################################################
async def get_user(request: Request):
    """
    Dependent function for getting user from access token
    """
    token = request.cookies.get("access_token")
    print("token = ", token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated_1")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # {'exp': 1748399039, 'sub': 'admin'}
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Not authenticated_2")

    except JWTError:
        raise HTTPException(status_code=401, detail="Not authenticated_3")

    user = username  # only for testing, need get func

    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated_4")

    return user


async def auth_check(user: str | None = Depends(get_user)):
    """
    additional auth check to get_user
    """
    sys.stderr.write("*** auth_check  ***\n")
    print(f"receiving {user} and return {user}")

    return user


# Custom middleware for authentication
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        sys.stderr.write("*** auth middleware  ***\n")
        # Skip auth for public routes
        if request.url.path in ["/static/logo.png", "/static/logo4.jpg", "/public", "/login", "/logout", "/protected", "/protected2"]:
            return await call_next(request)

        token = request.cookies.get("access_token")
        print("token = ", token)

        response = RedirectResponse(url="/login", status_code=303)

        if not token:
            return response

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # {'exp': 1748399039, 'sub': 'admin'}
            username: str = payload.get("sub")
            if username is None:
                return response

        except JWTError:
            return response

        session = Session(engine)
        user = session.query(User).filter(User.username == username).first()

        if user is None:
            return response

        # user exists but diabled
        if user.disabled:
            return templates.TemplateResponse("login.html", {"request": request, "error": "Disabled user"})

        print(f"user: {user} !!!")

        return await call_next(request)


# instante app
app = FastAPI()
app.add_middleware(AuthMiddleware)


# setting template / static dir at under current dir
app.mount(path="/static", app=StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# define func to convert df to SQL model
def vin_dataframe_to_sql(df, session):
    sys.stderr.write("*** vin df to SQL  ***\n")
    for index, row in df.iterrows():
        v = session.query(Vehicle).get(row['vin'])
        if not v:
            v = Vehicle()
            print("new vin")

        else:
            print("existing vin")

        v.vin = row['vin']
        print(row['date'])
        v.model = row['model']
        try:
            v.date = datetime.date.fromisoformat(row['date'])

        except Exception as e:
            v.date = datetime.date.today()
            print(e)

        v.odometer = row['odometer']
        v.status = row['status']
        session.add(v)

    session.commit()


################################################################################
# Start up and Shut down
################################################################################
# create tables at startup - there's separate module db_init.py as well
@app.on_event("startup")
def on_startup():
    sys.stderr.write("*** startup  ***\n")
    create_db_and_tables()


@app.on_event("shutdown")
def shutdown_event():
    sys.stderr.write("*** shutdown  ***\n")
    print("Database connections have been closed.")


################################################################################
# User auth
################################################################################
@app.get("/register", response_class=HTMLResponse)
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# user auth setting
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# maybe add 'avtive' field so that we can check users before let him/her in
@app.post("/register")
async def register_user(request: Request, session: SessionDep, username: str = Form(...), password: str = Form(...)):
    sys.stderr.write("*** register_user  ***\n")
    hashed_password = pwd_context.hash(password)
    user = User(username=username, hashed_password=hashed_password)
    session.add(user)
    session.commit()
    return RedirectResponse(url="/", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    """
    Only showing login page
    """
    return templates.TemplateResponse("login.html", {"request": request})


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/login")
async def login_user(response: Response, request: Request, session: SessionDep, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    form_data should have form_data.username & form_data.password
    username: demo
    password: test
    """
    sys.stderr.write("*** login_user  ***\n")
    user = session.query(User).filter(User.username == form_data.username).first()
    print(f"## Form data = {form_data.username} and the User = {user}")
##    user.hashed_password = pwd_context.hash("test")  # only for testing, override password to test
##    print(f"## password changed to {user.hashed_password}")


    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    # create access token, expire in 30 min
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=timedelta(minutes=30))

    # before return need to set cookie into redirect
##    return {"message": "Login successful"}  # for showing json
##    return templates.TemplateResponse("login.html", {"request": request, "success": "Login successful"})
    response = RedirectResponse(url="/", status_code=303)  # redirect

    # Set token in cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,  # Prevent JavaScript access
        max_age=1800,   # 30 minutes
        secure=True,    # Use HTTPS in production
        samesite="lax"
    )

    return response



@app.get("/protected")
def protected_route(user: str | None = Depends(get_user)):
    """
    Only for testing access token in cookie
    Not real security here yet
    """
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # You can decode and verify the token here
    return {"message": f"Access granted for {user}"}


@app.get("/protected2")
def protected_route2(user: str | None = Depends(auth_check)):
    """
    Only for testing access token in cookie with auth_check
    """
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # You can decode and verify the token here
    return {"message": f"Access granted for {user}"}


@app.get("/logout")
async def logout(response: Response, request: Request):
    """
    For logging out, functioning now
    """
    response = RedirectResponse(url="/login", status_code=303)  # redirect
    response.delete_cookie(key="access_token")
    return response




################################################################################
# Landing page
################################################################################
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    sys.stderr.write("*** index  ***\n")

    title = "APE(AGS + EVS) | with fastAPI"
    msg = "test message"
    flag = True
    note = "test note"
    alert = "Recall R1234 released"

    context = {
            "request": request,
            "title" : title,
            "message" : msg,
            "flag" : flag,
            "note" : note,
            "alert" : alert
        }

    return templates.TemplateResponse("index.html", context)


# Visit with GET like "/vin_search/?vin=VR1X25GGRM2802663"
@app.get("/vin_search/", response_class=HTMLResponse)
async def vin_serch(request: Request, session: SessionDep, vin: str = ""):
    sys.stderr.write("*** vin search  ***\n")
    vehicle = None
    msg = 'VIN search'

    if vin == "":
        context = {
            "request": request,
            "vin": vin,
            "vehicle" : vehicle,
            "message" : msg,
            "items" : [],
            }
        return templates.TemplateResponse("vin_search.html", context)

    # get empty record, without this get error - vehicle is not defined
##    vehicle = session.get(vehicle, 0)  # getting by id --> not foundd
    try:
        # for using 'WHERE LIKE' only for this method (first fetch)
        vin_like = "%{}%".format(vin)
        vehicle = session.query(Vehicle).filter(Vehicle.vin.like(vin_like)).one()
        print(vehicle)

        # overwite vin to full vin
        vin = vehicle.vin

        msg = "VIN found"
        items = session.exec(select(Ticket).where(Ticket.vin == vehicle.vin)).all()

    except Exception as e:
        msg = "VIN not found: {}".format(e)
        items = []

    context = {
            "request": request,
            "vin": vin,
            "vehicle" : vehicle,
            "message" : msg,
            "items" : items,
        }

    return templates.TemplateResponse("vin_search.html", context)


# for post method
# Redirected after vni_search/update
@app.post("/vin_search/", response_class=HTMLResponse)
async def vin_search_post(request: Request, session: SessionDep):
    sys.stderr.write("*** vin_search_post  ***\n")
    msg = 'VIN search'
    vehicle = Vehicle()
    context = {
            "request": request,
            "vehicle": vehicle,
            "message" : msg,
        }
    return templates.TemplateResponse("vin_search.html", context)


# for receiving /{}vin} in URL like having href
@app.get("/vin_search/{vin}", response_class=HTMLResponse)
async def vin_serch_get(request: Request, session: SessionDep, vin: str = ""):
    sys.stderr.write("*** vin_search_get  ***\n")
    msg = 'VIN search'

    vehicle = session.exec(select(Vehicle).where(Vehicle.vin == vin)).one()
    if not vehicle:
        vehicle = Vehicle()

    try:
        items = session.exec(select(Ticket).where(Ticket.vin == vehicle.vin)).all()

    except Exception as e:
        msg += ": {}".format(e)

    context = {
            "request": request,
            "vin": vin,
            "vehicle" : vehicle,
            "message" : msg,
            "items" : items,
        }

    return templates.TemplateResponse("vin_search.html", context)


@app.post('/vin_search/update', response_class=HTMLResponse)
async def update_vin_info(request: Request, session: SessionDep, vin: str = Form(), model: str = Form(), production_day: str = Form(), odometer: int = Form(), status: str = Form()):

    sys.stderr.write("*** update_vin_info  ***\n")
    try:
        vehicle = session.exec(select(Vehicle).where(Vehicle.vin == vin)).one()
        vehicle.model = model
        vehicle.odometer = odometer
        vehicle.status = status

        # need to convert to datetime, get error if not convertable to datetime
        try:
            vehicle.date = datetime.date.fromisoformat(production_day)

        except:
            vehicle.date = None

        session.add(vehicle)
        session.commit()
        session.refresh(vehicle)
##        session.close()
        msg = "vehicle data updated"


    except Exception as e:
        vehicle = Vehicle()
        msg = "Uodate failed: {}".format(e)


    context = {
            "vin": vin,
            "request": request,
            "vehicle": vehicle,
            "message" : msg,
        }

##    return RedirectResponse("/vin_search")  # for redirecting
    return templates.TemplateResponse("vin_search.html", context)


@app.post('/vin_search/new_ticket', response_class=HTMLResponse)
async def new_ticket(request: Request, session: SessionDep, vin: str = Form()):
    sys.stderr.write("*** new_ticket  ***\n")
    print(vin)  # only for testing

    try:
        # for exact match
        vehicle = session.exec(select(Vehicle).where(Vehicle.vin == vin)).one()

        # create ticket only if vhecile exicsts
        ticket = Ticket()
        ticket.vin = vin
        ticket.status = "Open"
        ticket.date_update = datetime.date.today()
        ticket.date_init = datetime.date.today()
        ticket.labour = 0
        session.add(ticket)
        session.commit()
##        session.close()
        items = session.exec(select(Ticket).where(Ticket.vin == vin)).all()
        msg = "New ticket was created"

    except Exception as e:
        vehicle = Vehicle()
        msg = "Failed to create ticket : {}".format(e)
        items = []

    context = {
            "vin": vin,
            "request": request,
            "vehicle": vehicle,
            "message" : msg,
            "items" : items,
        }

    return templates.TemplateResponse("vin_search.html", context)


################################################################################
# Ticket handlings
################################################################################
# for editing ticket, generating form dynamically with jinja template
@app.get("/ticket/edit/{id}", response_class=HTMLResponse)
async def ticket_edit(request: Request, session: SessionDep, id: int):
    sys.stderr.write("*** edit_ticket  ***\n")
    msg = "Editting {}".format(id)

    tkt = session.query(Ticket).get(id)
    tkt_dict = tkt.dict()

    # removing if from form field (No need to update id)
    tkt_dict.pop('id')

    # pop vin as well
    vin = tkt_dict.pop('vin')

    items = tkt_dict.items()

    # parts table

    # column labels
    parts_fields = Parts().dict().items()
    print("Look for id {}".format(id))
    parts_query = session.exec(select(Parts).where(Parts.ticket_id == id)).all()

    # create holder
    parts_list = []

    # create list of dict
    for p in parts_query:
        tmpDict = p.dict()
        tmpDict.pop('ticket_id')  # removing "ticket_id"
        parts_list.append(tmpDict)


## [[('order_id', 123456), ('ticket_id', 7), ('ref', '123345'), ('ata', datetime.date(2025, 4, 30)), ('cost', 550.0), ('id', 1), ('eta', datetime.date(2025, 6, 6)), ('tags', 'aa')], [('order_id', 234567), ('ticket_id', 7), ('ref', '23456'), ('ata', None), ('cost', 660.0), ('id', 2), ('eta', datetime.date(2025, 10, 12)), ('tags', 'bb')]]


    context = {
            "vin": vin,
            "id" : id,
            "request": request,
            "message": msg,
            "items" : items,
            "parts_fields" : parts_fields,
            "parts_list" : parts_list
        }
    return templates.TemplateResponse("edit_ticket.html", context)


# func for pressing 'submit' button
@app.post("/ticket/edit/{id}", response_class=HTMLResponse)
async def ticket_edit_update(request: Request, session: SessionDep, id: str,  status: str = Form(), date_update: datetime.date = Form(), date_init: datetime.date = Form(), labour: float | None = Form(), tags:str = Form()):
    sys.stderr.write("*** edit_ticket_update  ***\n")
    tkt = session.query(Ticket).get(id)
    try:
        tkt.status = status
        tkt.date_update = date_update
        tkt.date_init = date_init
        tkt.labour = labour
        tkt.tags = tags

        session.add(tkt)
        session.commit()
        session.refresh(tkt)

        tkt_dict = tkt.dict()

        # removing if from form field (No need to update id)
        tkt_dict.pop('id')

        vin = tkt_dict.pop('vin')
        items = tkt_dict.items()
        updated = True
        msg = "updated"

    except Exception as e:
        vin = ""
        items = [("Failed", "Failed")]
        updated = False
        msg = "Uodate failed: {}".format(e)

    # parts table
    # column labels
    parts_fields = Parts().dict().items()
    print("Look for id {}".format(id))
    parts_query = session.exec(select(Parts).where(Parts.ticket_id == id)).all()

    # create holder
    parts_list = []

    # create list of dict
    for p in parts_query:
        tmpDict = p.dict()
        tmpDict.pop('ticket_id')  # removing "ticket_id"
        parts_list.append(tmpDict)

    context = {
            "vin": vin,
            "request": request,
            "message": msg,
            "items" : items,
            "parts_fields" : parts_fields,
            "parts_list" : parts_list,
            "updated" : updated,
        }
    return templates.TemplateResponse("edit_ticket.html", context)


# for clicing 'back to vin search'
@app.get("/back_to_vin_search", response_class=HTMLResponse)
async def back_to_vin_search(request: Request, vin: str = ""):
    sys.stderr.write("*** back to vin search  ***\n")
    print(request.headers)

    return RedirectResponse(url="/vin_search/?vin={}".format(vin), status_code=303)


# for deleting ticket,
@app.get("/ticket/delete/{id}", response_class=HTMLResponse)
async def ticket_delete(request: Request, id: str, session: SessionDep):
    sys.stderr.write("*** delete_ticket  ***\n")

    tkt = session.query(Ticket).get(id)
    vin = ""

    try:
        vin = tkt.vin
        session.delete(tkt)
        session.commit()
        msg = "Deleted {}".format(id)

    except Exception as e:
        msg = "Failed to delete {0} for : {1}".format(id)

    return RedirectResponse(url="/vin_search/?vin={}".format(vin), status_code=303)


################################################################################
# Parts
################################################################################
# func for pressing 'submit' button
@app.get("/parts/edit/{id}", response_class=HTMLResponse)
async def parts_edit(request: Request, session: SessionDep, id: str):
    sys.stderr.write("*** edit_parts  ***\n")
##    referer = request.headers['referer']  # for returning to previous page, not used for now
    msg = "edit : {}".format(id)
    pts = session.query(Parts).get(id)
    pts_dict = pts.dict()

    # removing if from form field (No need to update id)
    pts_dict.pop('id')
    items = pts_dict.items()

    # for general edit
    context = {
            "id": id,
            "request": request,
            "message": msg,
            "items" : items,
            "updated" : True,  # set True if want always be
        }

    return templates.TemplateResponse("edit_general.html", context)


@app.post('/parts/new_parts', response_class=HTMLResponse)
async def new_ticket(request: Request, session: SessionDep, id: int = Form()):
    sys.stderr.write("*** new_parts  ***\n")
    print(id)  # only for testing

    try:
        # create ticket only if vhecile exicsts
        parts = Parts()
        parts.ticket_id = id
        session.add(parts)
        session.commit()
        msg = "New parts was created"

    except Exception as e:
        parts = Parts()
        msg = "Failed to create parts : {}".format(e)

    return RedirectResponse(url="/ticket/edit/{}".format(id), status_code=303)


# func for pressing 'submit' button
@app.post("/parts/edit/{id}", response_class=HTMLResponse)
async def ticket_edit_update(request: Request, session: SessionDep, id: int,  order_id: int | None = Form(), ticket_id: int = Form(), ref : str = Form(), eta: datetime.date | None = Form(), ata: datetime.date | None = Form(), cost: float | None = Form(), tags:str = Form()):
    sys.stderr.write("*** edit_parts_update  ***\n")
    pts = session.query(Parts).get(id)
    pts.order_id = order_id
    pts.ticket_id = ticket_id
    pts.ref = ref
    print("eta : ", eta)
    pts.eta = eta
    pts.ata = ata
    pts.cost = cost
    pts.tags = tags

    session.add(pts)
    session.commit()
    session.refresh(pts)

    return RedirectResponse(url="/parts/edit/{}".format(id), status_code=303)



################################################################################
# Camera
################################################################################
# Camera
@app.get('/camera', response_class=HTMLResponse)
async def photo_index(request: Request):
    sys.stderr.write("*** camera  ***\n")
    context = {
            "request": request,
        }

    return templates.TemplateResponse("camera.html", context)


@app.post("/upload-photo/")
async def upload_photo(file: UploadFile = File(...)):
    file_location = f"photos/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}




################################################################################
# Export
###############################################################################
# export file, filter car, and show in pages
@app.get("/export")
async def export_index(request: Request, session: SessionDep, skip: int = 0, limit: int = 3, ):
    vehicles = session.query(Vehicle).offset(skip).limit(limit).all()
    msg = "Export test"
    context = {
            "request": request,
            "message": msg,
            "vehicles": vehicles,
            "skip": skip,
            "limit": limit,
        }
    return templates.TemplateResponse("export.html", context)


# for exporting csv by route
# route is maybe for "Get" + "POST"
@app.route("/export/download_csv", methods=["GET", "POST"])
async def download_csv(request: Request):
    dframe_v = pd.read_sql('select * from vehicle', engine)
    dframe_v.to_csv("vhecile_all.csv", index = False, encoding = "utf_8_sig")
    return FileResponse("vhecile_all.csv", filename="vhecile_all_1.csv")


################################################################################
# Import
################################################################################
# for importing csv file
@app.get("/import")
async def import_index(request: Request, session: SessionDep):
    msg = "Import test"
    context = {
            "request": request,
            "message": msg,
        }
    return templates.TemplateResponse("import.html", context)


@app.post("/upload")
async def upload_any_file(request: Request):
    sys.stderr.write("*** upload any file  ***\n")
    form = await request.form()
    file = form['ufile']
    bf = await file.read()
    with open(file.filename,'wb') as f:
        f.write(bf)
    return RedirectResponse('/import',status_code=301)


# for drag & drop
@app.post("/upload-csv/")
async def upload_csv_1(file: UploadFile = File(...)):
    sys.stderr.write("*** upload_csv_1 ***\n")
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        expected_columns = {"vin", "odometer"}
        if not expected_columns.issubset(df.columns):
            return JSONResponse(status_code=400, content={"detail": "missing columns"})

        print(df)
        return {"message": "upladed"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


# from csv to SQL
@app.post("/from_csv")
async def upload_csv_2(csvFile: UploadFile = File(...), session: SessionDep = None):
    sys.stderr.write("*** upload_csv_2 ***\n")
    _csvdata = StringIO(str(csvFile.file.read(), 'utf-8'))

    try:
        # pandas dataframe
        df = pd.read_csv(_csvdata)

        # check requred columns
        expected_columns = {"vin", "odometer"}
        if not expected_columns.issubset(df.columns):
            raise HTTPException(status_code=400, detail="CSV does not contain requred field")

        # get error with date column, need to work with datetime
        # only append not able to update existing record
##        df.to_sql("vehicle", con=engine, if_exists="append", index=False)

        # use separate func
        print(session)
        vin_dataframe_to_sql(df, session)

        return RedirectResponse('/import',status_code=301)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during importing: {str(e)}")


# not working
# js does not send 'file-name' header
@app.post('/feedFile')
async def upload_with_header(request: Request):
    sys.stderr.write("*** upload with header  ***\n")
    bf = await request.body()
    print(request.headers)
    name = request.headers['file-name']
    with open(name,'wb') as f:
        f.write(bf)

    return RedirectResponse('/import',status_code=301)


# test for getting item from get
@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="item.html", context={"id": id}
    )


################################################################################
# HTMX test
################################################################################
# for importing csv file
@app.get("/htmx_test")
async def load_htmx_page(request: Request, session: SessionDep):
    sys.stderr.write("*** load HTMX page ***\n")
    msg = "HTMX Test"
    items = session.exec(select(NoteItem)).all()
    context = {
            "request": request,
            "message" : msg,
            "items": items,
        }
    return templates.TemplateResponse("htmx_test.html", context)


@app.post("/htmx_test/add")
async def add_note(request: Request, session: SessionDep):
    sys.stderr.write("*** HTMX/add ***\n")
##    try:
    form = await request.form()
    item = form.get("item")
    note_item_create = NoteItemCreate(item=item)
    note_item = NoteItem(item=note_item_create.item)
    session.add(note_item)
    session.commit()
##    except ValueError as e:
##        raise HTTPException(status_code=400, detail=str(e))
    items = session.exec(select(NoteItem)).all()
    context = {
            "request": request,
            "items": items,
        }
    return templates.TemplateResponse("note_list.html", context)


@app.post("/htmx_test/update/{id}")
async def update_note(id: int, request: Request, session: SessionDep):
    sys.stderr.write("*** uodate ***\n")
    form = await request.form()
    item = form.get("item")
    note_item = session.get(NoteItem, id)
    if note_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    note_item.item = item
    session.add(note_item)
    session.commit()
    items = session.exec(select(NoteItem)).all()
    context = {
            "request": request,
            "items": items,
        }
    return templates.TemplateResponse("note_list.html", context)


@app.post("/htmx_test/delete/{id}")
async def del_note(id: int, request: Request, session: SessionDep):
    sys.stderr.write("*** delete ***\n")
    note_item = session.get(NoteItem, id)
    if note_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    session.delete(note_item)
    session.commit()
    items = session.exec(select(NoteItem)).all()
    context = {
            "request": request,
            "items": items,
        }
    return templates.TemplateResponse("note_list.html", context)


################################################################################
# Report
################################################################################
# Report
@app.get('/report', response_class=HTMLResponse)
async def report_index(request: Request):
    sys.stderr.write("*** report  ***\n")
    context = {
            "request": request,
        }

    return templates.TemplateResponse("report.html", context)


@app.get("/report/test_pdf")
async def generate_pdf():
    sys.stderr.write("*** generate_pdf  ***\n")

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=portrait(A4))

    # Add content to the PDF
    p.drawString(440, 790, "Date: {}".format(datetime.date.today()))
    p.drawImage("static/APE.Logo.png", 40, 760)
    p.drawString(100, 720, "Hello, this is a PDF generated with ReportLab and FastAPI!")
    p.drawString(100, 700, "You can customize this content as needed.")

    p.showPage()
    p.save()

    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=test.pdf"
    })



# only for testing models
if __name__ == '__main__':
    print(__name__)

    # create SQL table
    SQLModel.metadata.create_all(engine)
    db_tables = SQLModel.metadata.tables
    print(("*** tables  ***"))
    print(dict(db_tables))

    print(("*** session  ***"))
    engine = create_engine(sqlite_url, connect_args=connect_args)
    session = Session(engine)

    # get vhecile qyery v.all() for list
    v = session.query(Vehicle)

    # get pandas data frame --> generate csv
    dframe_v = pd.read_sql('select * from vehicle', engine)
    dframe_v.to_csv("fastAPI_test.csv", index = False, encoding = "utf_8_sig")

    vin = "ZARPATDW7N3046433"







