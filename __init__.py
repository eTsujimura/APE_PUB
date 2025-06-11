#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      U640919
#
# Created:     28/03/2025
# Copyright:   (c) Ei Tsujimujra 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import uvicorn
import main

if __name__ == '__main__':
  uvicorn.run('main:app', host='127.0.0.1', port=8080, log_level='info', reload=False)
##  uvicorn.run('sqlmodel_test:app', host='127.0.0.1', port=8080, log_level='info', reload=True)